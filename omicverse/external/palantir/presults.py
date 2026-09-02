from typing import Union, Optional, List, Dict
import numpy as np
import pandas as pd
import pickle
import time
import importlib.util
import sys

from collections import OrderedDict

from sklearn.preprocessing import StandardScaler
import scanpy as sc
from anndata import AnnData

from . import config
from .validation import _validate_obsm_key, _validate_varm_key
from ..._anndata_compat import as_dtype as _as_dtype


# Used for trend computation and branch selection
PSEUDOTIME_RES = 500


class PResults(object):
    """
    Container of palantir results
    """

    def __init__(self, pseudotime, entropy, branch_probs, waypoints):
        # Initialize
        self._pseudotime = pseudotime
        self._entropy = entropy
        self._branch_probs = branch_probs
        self._branch_probs[self._branch_probs < 0.01] = 0
        self._waypoints = waypoints

    # Getters and setters
    @property
    def pseudotime(self):
        return self._pseudotime

    @property
    def branch_probs(self):
        return self._branch_probs

    @branch_probs.setter
    def branch_probs(self, branch_probs):
        self._branch_probs = branch_probs

    @property
    def entropy(self):
        return self._entropy

    @entropy.setter
    def entropy(self, entropy):
        self._entropy = entropy

    @property
    def waypoints(self):
        return self._waypoints

    @classmethod
    def load(cls, pkl_file):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        # Set up object
        presults = cls(
            data["_pseudotime"],
            data["_entropy"],
            data["_branch_probs"],
            data["_waypoints"],
        )
        return presults

    def save(self, pkl_file: str):
        pickle.dump(vars(self), pkl_file)


def compute_gene_trends_legacy(
    data: Union[AnnData, PResults],
    gene_exprs: Optional[pd.DataFrame] = None,
    lineages: Optional[List[str]] = None,
    n_splines: int = 4,
    spline_order: int = 2,
    n_jobs: int = -1,
    expression_key: str = "MAGIC_imputed_data",
    pseudo_time_key: str = "palantir_pseudotime",
    fate_prob_key: str = "palantir_fate_probabilities",
    gene_trend_key: str = "palantir_gene_trends",
    save_as_df: bool = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Computes gene expression trends along pseudotemporal trajectories.

    This function calculates gene expression trends and their standard deviations
    along pseudotemporal trajectories computed by Palantir.

    Parameters
    ----------
    data : Union[AnnData, palantir.presults.PResults]
        Either a Scanpy AnnData object or a Palantir results object.
    gene_exprs : pd.DataFrame, optional
        DataFrame of gene expressions with cells as rows and genes as columns. If not provided, gene expressions
        will be fetched from `data` using `expression_key`.
    lineages : List[str], optional
        List of lineages for which to compute the trends. If None, all columns of the fate probability matrix are used. Default is None.
    n_splines : int, optional
        Number of splines to use. Must be non-negative. Default is 4.
    spline_order : int, optional
        Order of the splines to use. Must be non-negative. Default is 2.
    n_jobs : int, optional
        Number of cores to use. If -1, all available cores will be used. Default is -1.
    expression_key : str, optional
        Key to access gene expression matrix from a layer of the AnnData object. Default is 'MAGIC_imputed_data'.
        Ignored if `gene_exprs` is provided.
    pseudo_time_key : str, optional
        Key to access pseudotime from the obs attribute of the AnnData object. Default is 'palantir_pseudotime'.
    fate_prob_key : str, optional
        Key to access fate probabilities from the obsm attribute of the AnnData object. Default is 'palantir_fate_probabilities'.
    gene_trend_key : str, optional
        Key to store the computed gene trends in the varm attribute of the AnnData object. Default is 'palantir_gene_trends'.
        The trends for each lineage are stored under 'varm[gene_trend_key + "_" + lineage_name]'.
        The pseudotime points at which the trends are computed are stored in the uns attribute
        under 'uns[gene_trend_key + "_" + lineage_name + "_pseudotime"]'.
    save_as_df : bool, optional
        If True, the trends will be saved in the varm of the AnnData object as pandas DataFrame with pseudotime as column names.
        If False, the trends will be saved as numpy array and the pseudotime in uns[gene_trend_key + "_" + lineage_name + "_pseudotime"].
        The option to save as DataFrame is there due to some versions of AnnData not being able to
        write h5ad files with DataFrames in ad.varm. Default is palantir.SAVE_AS_DF=True.

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Dictionary of gene expression trends and standard deviations for each branch.
    """
    from joblib import delayed, Parallel
    if save_as_df is None:
        save_as_df = config.SAVE_AS_DF

    # Extract palantir results from AnnData if necessary
    if isinstance(data, AnnData):
        gene_exprs = data.to_df(expression_key)
        pseudo_time = pd.Series(data.obs[pseudo_time_key], index=data.obs_names)
        fate_probs, _ = _validate_obsm_key(data, fate_prob_key)

        pr_res = PResults(pseudo_time, None, fate_probs, None)
    else:
        pr_res = data

    # Compute for all lineages if branch is not speicified
    if lineages is None:
        lineages = pr_res.branch_probs.columns

    # Results Container
    results = OrderedDict()
    for branch in lineages:
        results[branch] = OrderedDict()
        # Bins along the pseudotime
        br_cells = pr_res.branch_probs.index[pr_res.branch_probs.loc[:, branch] > 0.7]
        bins = np.linspace(0, pr_res.pseudotime[br_cells].max(), PSEUDOTIME_RES)

        # Branch results container
        results[branch]["trends"] = pd.DataFrame(0.0, index=gene_exprs.columns, columns=bins)
        results[branch]["std"] = pd.DataFrame(0.0, index=gene_exprs.columns, columns=bins)

    # Compute for each branch
    for branch in lineages:
        print(branch)
        start = time.time()

        # Branch cells and weights
        weights = pr_res.branch_probs.loc[gene_exprs.index, branch].values
        bins = np.array(results[branch]["trends"].columns)
        res = Parallel(n_jobs=n_jobs)(
            delayed(gam_fit_predict)(
                pr_res.pseudotime[gene_exprs.index].values,
                gene_exprs.loc[:, gene].values,
                weights,
                bins,
                n_splines,
                spline_order,
            )
            for gene in gene_exprs.columns
        )

        # Fill in the matrices
        for i, gene in enumerate(gene_exprs.columns):
            results[branch]["trends"].loc[gene, :] = res[i][0]
            results[branch]["std"].loc[gene, :] = res[i][1]
        if isinstance(data, AnnData):
            if save_as_df:
                df = results[branch]["trends"]
                df.columns = df.columns.astype(str)
                data.varm[gene_trend_key + "_" + branch] = df
            else:
                data.varm[gene_trend_key + "_" + branch] = results[branch]["trends"].values
                data.uns[gene_trend_key + "_" + branch + "_pseudotime"] = results[branch][
                    "trends"
                ].columns.values
        end = time.time()
        print("Time for processing {}: {} minutes".format(branch, (end - start) / 60))

    return results


def compute_gene_trends(
    ad: AnnData,
    lineages: Optional[List[str]] = None,
    masks_key: str = "branch_masks",
    expression_key: str = None,
    pseudo_time_key: str = "palantir_pseudotime",
    gene_trend_key: str = "gene_trends",
    save_as_df: bool = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Compute gene expression trends along pseudotime in the given AnnData object.

    This function computes the gene expression trends for each branch of the
    pseudotime trajectory using mellon.FunctionEstimator. The computed gene
    trends are stored in the varm attribute of the AnnData object, with keys
    in the format '{gene_trend_key}_{branch}'. Each key maps to a 2D numpy
    array where rows correspond to genes and columns correspond to the
    pseudotime grid. The pseudotime grid for each branch is stored in the uns
    attribute of the AnnData object, with keys in the format
    '{gene_trend_key}_{branch}_pseudotime'.

    Parameters
    ----------
    ad : AnnData
        AnnData object containing the gene expression data and pseudotime.
    lineages : List[str], optional
        Subset of lineages for which to compute the trends.
        If None uses all columns of the fate probability matrix.
        Default is None.
    masks_key : str, optional
        Key to access the branch cell selection masks from obsm of the AnnData object.
        Default is 'branch_masks'.
    expression_key : str, optional
        Key to access the gene expression data in the layers of the AnnData object.
        If None, uses raw expression data in .X. Default is None.
    pseudo_time_key : str, optional
        Key to access the pseudotime values in the AnnData object. Default is 'palantir_pseudotime'.
    gene_trend_key : str, optional
        Key base to store the gene expression trends in varm of the AnnData object.
        Default is 'palantir_gene_trends'.
    save_as_df : bool, optional
        If True, the trends will be saved in the varm of the AnnData object as pandas DataFrame with pseudotime as column names.
        If False, the trends will be saved as numpy array and the pseudotime in uns[gene_trend_key + "_" + lineage_name + "_pseudotime"].
        The option to save as DataFrame is there due to some versions of AnnData not being able to
        write h5ad files with DataFrames in ad.varm. Default is palantir.SAVE_AS_DF=True.
    **kwargs
        Additional arguments to be passed to mellon.FunctionEstimator.

    Returns
    -------
    dict
        A dictionary containing gene expression trends for each branch. The keys of the dictionary
        are the branch names. The value for each branch is a sub-dictionary with a key 'trends' that
        maps to a DataFrame. The DataFrame contains the gene expression trends, indexed by gene names
        and columns representing pseudotime points.
    """
    if save_as_df is None:
        save_as_df = config.SAVE_AS_DF
    # Check the AnnData object for the necessary keys
    if pseudo_time_key not in ad.obs_keys():
        raise ValueError(f"'{pseudo_time_key}' is not found in the AnnData object's obs.")

    masks, branches = _validate_obsm_key(ad, masks_key, as_df=False)

    gene_exprs = ad.to_df(expression_key)
    pseudo_time = ad.obs[pseudo_time_key].values

    if lineages is not None:
        for lin in lineages:
            if lin not in branches:
                raise ValueError(
                    f"Lineage '{lin}' does not seem to have a selection in obsm['{masks_key}']."
                )
    else:
        lineages = branches

    # Import mellon locally to avoid JAX fork warnings in other parts of the code
    import mellon
    
    # Set the default arguments for mellon.FunctionEstimator
    mellon_args = dict(sigma=1, ls=1)

    lagacy_results = dict()
    # Compute the gene expression trends
    for i, branch in enumerate(branches):
        if branch not in lineages:
            continue
        print(branch)
        mask = masks[:, i]
        pt = pseudo_time[mask]
        pt_grid = np.linspace(pt.min(), pt.max(), PSEUDOTIME_RES)
        expr = gene_exprs.loc[mask, :]
        mellon_args["landmarks"] = pt_grid
        mellon_args.update(kwargs)
        func_est = mellon.FunctionEstimator(**mellon_args)
        result = func_est.fit_predict(pt, expr, pt_grid).T
        result = np.asarray(result)

        lagacy_results[branch] = {
            "trends": pd.DataFrame(result, columns=pt_grid, index=ad.var_names)
        }

        if save_as_df:
            ad.varm[gene_trend_key + "_" + branch] = pd.DataFrame(
                result,
                columns=pt_grid.astype(str),
                index=ad.var_names,
            )
        else:
            ad.varm[gene_trend_key + "_" + branch] = result
            ad.uns[gene_trend_key + "_" + branch + "_pseudotime"] = pt_grid

    return lagacy_results


def gam_fit_predict(x, y, weights=None, pred_x=None, n_splines=4, spline_order=2):
    """
    Function to compute individual gene trends using pyGAM

    This function requires the optional pygam package. If not installed, it will raise an
    ImportError with instructions on how to install it.

    Parameters
    ----------
    x : array-like
        Pseudotime axis
    y : array-like
        Magic imputed expression for one gene
    weights : array-like, optional
        Lineage branch weights
    pred_x : array-like, optional
        Pseudotime axis for predicted values
    n_splines : int, optional
        Number of splines to use. Must be non-negative.
    spline_order : int, optional
        Order of spline to use. Must be non-negative.

    Returns
    -------
    tuple
        Predicted values and their standard deviations.

    Raises
    ------
    ImportError
        If pygam is not installed. Install with `pip install pygam` or
        `pip install palantir[gam]`.
    """
    try:
        from pygam import LinearGAM, s
    except ImportError:
        raise ImportError(
            "The pygam package is required for this function but not installed. "
            "You can install it with:\n"
            "    pip install pygam\n"
            "or:\n"
            "    pip install palantir[gam]"
        )

    # Check for known compatibility issues
    try:
        import scipy.sparse as sp

        test_matrix = sp.csr_matrix((1, 1))
        if not hasattr(test_matrix, "A"):
            import warnings

            warnings.warn(
                "Your scipy version may be incompatible with the installed pygam version. "
                "If you encounter errors, consider using an older scipy version or updating pygam."
            )
    except:
        pass  # Silently ignore any errors in the compatibility check

    # Weights
    if weights is None:
        weights = np.repeat(1.0, len(x))

    # Construct dataframe
    use_inds = np.where(weights > 0)[0]

    # GAM fit
    gam = LinearGAM(s(0, n_splines=n_splines, spline_order=spline_order)).fit(
        x[use_inds], y[use_inds], weights=weights[use_inds]
    )

    # Predict
    if pred_x is None:
        pred_x = x
    y_pred = gam.predict(pred_x)

    # Standard deviations
    p = gam.predict(x[use_inds])
    n = len(use_inds)
    sigma = np.sqrt(((y[use_inds] - p) ** 2).sum() / (n - 2))
    stds = (
        np.sqrt(1 + 1 / n + (pred_x - np.mean(x)) ** 2 / ((x - np.mean(x)) ** 2).sum()) * sigma / 2
    )

    return y_pred, stds


def _gam_fit_predict_rpy2(x, y, weights=None, pred_x=None):
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, Formula
    from rpy2.robjects.packages import importr

    pandas2ri.activate()

    # Weights
    if weights is None:
        weights = np.repeat(1.0, len(x))

    # Construct dataframe
    use_inds = np.where(weights > 0)[0]
    r_df = pandas2ri.py2rpy(pd.DataFrame(np.array([x, y]).T[use_inds, :], columns=["x", "y"]))

    # Fit the model
    rgam = importr("gam")
    model = rgam.gam(Formula("y~s(x)"), data=r_df, weights=pd.Series(weights[use_inds]))

    # Predictions
    if pred_x is None:
        pred_x = x
    y_pred = np.array(
        robjects.r.predict(model, newdata=pandas2ri.py2rpy(pd.DataFrame(pred_x, columns=["x"])))
    )

    # Standard deviations
    p = np.array(
        robjects.r.predict(
            model, newdata=pandas2ri.py2rpy(pd.DataFrame(x[use_inds], columns=["x"]))
        )
    )
    n = len(use_inds)
    sigma = np.sqrt(((y[use_inds] - p) ** 2).sum() / (n - 2))
    stds = (
        np.sqrt(1 + 1 / n + (pred_x - np.mean(x)) ** 2 / ((x - np.mean(x)) ** 2).sum()) * sigma / 2
    )

    return y_pred, stds


def cluster_gene_trends(
    data: Union[AnnData, pd.DataFrame],
    branch_name: str,
    genes: Optional[List[str]] = None,
    gene_trend_key: Optional[str] = "gene_trends",
    n_neighbors: int = 150,
    **kwargs,
) -> pd.Series:
    """
    Cluster gene trends using the Leiden algorithm.

    This function applies the Leiden clustering algorithm to gene expression trends
    along the pseudotemporal trajectory. If the input is an AnnData object, it uses
    the gene trends stored in the `varm` attribute accessed using the `gene_trend_key`.
    If the input is a DataFrame, it directly uses the input data for clustering.

    Parameters
    ----------
    data : Union[AnnData, pd.DataFrame]
        AnnData object or a DataFrame of gene expression trends.
    branch_name : str
        Name of the branch for which the gene trends are to be clustered.
    genes : list of str, optional
        List of genes to be considered for clustering. If None, all genes are considered. Default is None.
    gene_trend_key : str, optional
        Key to access gene trends in the AnnData object's varm. Default is 'palantir_gene_trends'.
    n_neighbors : int, optional
        The number of nearest neighbors to use for the k-NN graph construction. Default is 150.
    **kwargs
        Additional keyword arguments passed to `scanpy.tl.leiden`.

    Returns
    -------
    pd.Series
        A pandas series with the cluser lables for all passed genes.

    Raises
    ------
    KeyError
        If `gene_trend_key` is None when `data` is an AnnData object.
    """
    if isinstance(data, AnnData):
        if gene_trend_key is None:
            raise KeyError("Must provide a gene_trend_key when data is an AnnData object.")
        trends, pseudotimes = _validate_varm_key(data, gene_trend_key + "_" + branch_name)
    else:
        trends = data

    if genes is not None:
        trends = trends.loc[genes, :]

    # Standardize the trends
    trends = pd.DataFrame(
        StandardScaler().fit_transform(trends.T).T,
        index=trends.index,
        columns=trends.columns,
    )

    gt_ad = AnnData(_as_dtype(trends.values, np.float32))
    sc.pp.neighbors(gt_ad, n_neighbors=n_neighbors, use_rep="X")
    
    # Add required kwargs for leiden with igraph backend to avoid FutureWarning
    leiden_kwargs = {'flavor': 'igraph', 'n_iterations': 2, 'directed': False}
    # Override with user-provided kwargs if any
    leiden_kwargs.update(kwargs)
    
    sc.tl.leiden(gt_ad, **leiden_kwargs)

    communities = pd.Series(gt_ad.obs["leiden"].values, index=trends.index)

    if isinstance(data, AnnData):
        col_name = gene_trend_key + "_clusters"
        if genes is None:
            data.var[col_name] = communities
        else:
            data.var[col_name] = communities
            data.var[col_name] = data.var[col_name].astype("category")

    return communities


def select_branch_cells(
    ad: AnnData,
    pseudo_time_key: str = "palantir_pseudotime",
    fate_prob_key: str = "palantir_fate_probabilities",
    q: float = 1e-2,
    eps: float = 1e-2,
    masks_key: str = "branch_masks",
    save_as_df: bool = None,
):
    """
    Selects cells along specific branches of pseudotime ordering.

    This function identifies cells that are most likely to follow a certain lineage or "fate" by looking at
    their pseudotime order and fate probabilities. These cells are expected to be along the path of differentiation
    towards that specific fate.

    Parameters
    ----------
    ad : AnnData
        Annotated data matrix. The pseudotime and fate probabilities should be stored under the keys provided.
    pseudo_time_key : str, optional
        Key to access the pseudotime from obs of the AnnData object. Default is 'palantir_pseudotime'.
    fate_prob_key : str, optional
        Key to access the fate probabilities from obsm of the AnnData object. Default is 'palantir_fate_probabilities'.
    q : float, optional
        Quantile used to determine the threshold for the fate probability.
        This parameter should be between 0 and 1. Default is 1e-2.
    eps : float, optional
        A small constant substracted from the fate probability threshold. Default is 1e-2.
    masks_key : str, optional
        Key under which the resulting branch cell selection masks are stored in the obsm of the AnnData object.
        Default is 'branch_masks'.
    save_as_df : bool, optional
        If True, the masks will be saved in obsm of the AnnData object as pandas DataFrame. If False, the masks will be
        saved as numpy array. The option to save as numpy array is there due to some versions of AnnData not being able to
        write h5ad files with DataFrames in ad.obsm. Default is palantir.SAVE_AS_DF.

    Returns
    -------
    masks : np.ndarray
        An array of boolean masks that indicates whether each cell is on the path to each fate.
    """

    if save_as_df is None:
        save_as_df = config.SAVE_AS_DF

    # make sure that the necessary keys are in the AnnData object
    if pseudo_time_key not in ad.obs:
        raise KeyError(f"{pseudo_time_key} not found in ad.obs")

    fate_probs, fate_names = _validate_obsm_key(ad, fate_prob_key, as_df=False)
    pseudotime = ad.obs[pseudo_time_key].values

    fate_probs[np.isnan(fate_probs)] = 1 / fate_probs.shape[1]

    idx = np.argsort(pseudotime)
    sorted_fate_probs = fate_probs[idx, :]
    prob_thresholds = np.empty_like(fate_probs)
    n = fate_probs.shape[0]

    pseudotime_resolution = min(PSEUDOTIME_RES, n)

    step = n // pseudotime_resolution
    nsteps = n // step
    for i in range(nsteps):
        l, r = i * step, (i + 1) * step
        mprob = np.quantile(sorted_fate_probs[:r, :], 1 - q, axis=0)
        prob_thresholds[l:r, :] = mprob[None, :]
    mprob = np.quantile(sorted_fate_probs, 1 - q, axis=0)
    prob_thresholds[r:, :] = mprob[None, :]

    prob_thresholds = np.maximum.accumulate(prob_thresholds, axis=0)

    masks = np.empty_like(fate_probs).astype(bool)
    masks[idx, :] = prob_thresholds - eps < sorted_fate_probs

    if save_as_df:
        ad.obsm[masks_key] = pd.DataFrame(
            masks,
            columns=fate_names,
            index=ad.obs_names,
        )
    else:
        ad.obsm[masks_key] = masks
        ad.uns[masks_key + "_columns"] = fate_names

    return masks
