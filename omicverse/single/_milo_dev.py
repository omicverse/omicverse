from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
#import patsy
#from inmoose import edgepy

if TYPE_CHECKING:
    from collections.abc import Sequence
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances

from .._settings import Colors


def _log_info(message: str) -> None:
    print(f"{Colors.BLUE}{message}{Colors.ENDC}")


def _log_warning(message: str) -> None:
    print(f"{Colors.WARNING}{message}{Colors.ENDC}")


def _log_error(message: str) -> None:
    print(f"{Colors.FAIL}{message}{Colors.ENDC}")


def _calcFactorTMM(obs, ref, obs_lib_size, ref_lib_size,
                   logratioTrim=0.3, sumTrim=0.05, doWeighting=True, Acutoff=-1e10):
    """
    Calculate TMM factor for one sample against reference.

    This implements the core TMM (Trimmed Mean of M-values) algorithm
    as used in edgeR's calcNormFactors.
    """
    # Remove genes with zero counts in both samples
    nzobs = obs > 0
    nzref = ref > 0
    keep = nzobs & nzref

    if keep.sum() == 0:
        return 1.0

    obs = obs[keep]
    ref = ref[keep]

    # Calculate M values (log ratios) and A values (average log expression)
    obs_prop = obs / obs_lib_size
    ref_prop = ref / ref_lib_size
    M = np.log2(obs_prop / ref_prop)
    A = (np.log2(obs_prop) + np.log2(ref_prop)) / 2

    # Remove infinite values
    finite = np.isfinite(M) & np.isfinite(A)
    M = M[finite]
    A = A[finite]
    obs = obs[finite]
    ref = ref[finite]

    if len(M) == 0:
        return 1.0

    # Remove genes with A values below cutoff
    if Acutoff > -1e10:
        keep = A > Acutoff
        M, A, obs, ref = M[keep], A[keep], obs[keep], ref[keep]

    if len(M) == 0:
        return 1.0

    # Trim by M and A
    if logratioTrim > 0:
        M_cutoff_low = np.percentile(M, logratioTrim * 100)
        M_cutoff_high = np.percentile(M, (1 - logratioTrim) * 100)
        keep_M = (M >= M_cutoff_low) & (M <= M_cutoff_high)
    else:
        keep_M = np.ones(len(M), dtype=bool)

    if sumTrim > 0:
        A_cutoff_low = np.percentile(A, sumTrim * 100)
        A_cutoff_high = np.percentile(A, (1 - sumTrim) * 100)
        keep_A = (A >= A_cutoff_low) & (A <= A_cutoff_high)
    else:
        keep_A = np.ones(len(A), dtype=bool)

    keep = keep_M & keep_A
    M, A, obs, ref = M[keep], A[keep], obs[keep], ref[keep]

    if len(M) == 0:
        return 1.0

    # Calculate weights
    if doWeighting:
        weights = 1.0 / (1.0/obs + 1.0/ref)
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(M)) / len(M)

    # Weighted mean of M
    tmm = 2 ** np.sum(weights * M)

    return tmm


def calcNormFactors(counts, method="TMM", ref_column=None, logratioTrim=0.3, sumTrim=0.05,
                    doWeighting=True, Acutoff=-1e10):
    """
    Calculate normalization factors using TMM method.

    This is a Python implementation of edgeR's calcNormFactors function.

    Parameters:
    -----------
    counts : np.ndarray
        Count matrix (features x samples)
    method : str
        Normalization method ("TMM" or "none")
    ref_column : int
        Reference sample column index. If None, use sample closest to mean
    logratioTrim : float
        Amount of trim to use on log-ratios (M-values)
    sumTrim : float
        Amount of trim to use on combined absolute levels (A-values)
    doWeighting : bool
        Whether to compute weights
    Acutoff : float
        Cutoff on A-values to use before trimming

    Returns:
    --------
    norm_factors : np.ndarray
        Normalization factors for each sample
    """
    counts = np.asarray(counts, dtype=float)
    n_features, n_samples = counts.shape

    if method == "none":
        return np.ones(n_samples)

    if method != "TMM":
        raise ValueError(f"Only TMM method is supported, got: {method}")

    # Calculate library sizes
    lib_sizes = counts.sum(axis=0)

    # Find reference column (sample closest to mean)
    if ref_column is None:
        f = counts / lib_sizes
        geom_means = np.exp(np.mean(np.log(f + 1e-10), axis=0))
        ref_column = np.argmin(np.abs(geom_means - np.median(geom_means)))

    # Calculate TMM factors
    norm_factors = np.ones(n_samples)
    ref_counts = counts[:, ref_column]
    ref_lib_size = lib_sizes[ref_column]

    for i in range(n_samples):
        if i == ref_column:
            continue

        obs_counts = counts[:, i]
        obs_lib_size = lib_sizes[i]

        norm_factors[i] = _calcFactorTMM(
            obs_counts, ref_counts,
            obs_lib_size, ref_lib_size,
            logratioTrim=logratioTrim,
            sumTrim=sumTrim,
            doWeighting=doWeighting,
            Acutoff=Acutoff
        )

    # Normalize factors so their geometric mean is 1
    norm_factors = norm_factors / np.exp(np.mean(np.log(norm_factors)))

    return norm_factors


class Milo:
    """Python implementation of Milo."""

    def __init__(self):
        # No need to import R packages anymore
        pass

    def load(
        self,
        input: AnnData,
        feature_key: str | None = "rna",
    ):
        """Prepare a MuData object for subsequent processing.

        Args:
            input: AnnData
            feature_key: Key to store the cell-level AnnData object in the MuData object

        Returns:
            :class:`mudata.MuData` object with original AnnData.

        Examples:
            >>> import omicverse as ov
            >>> adata = ov.datasets.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)

        """
        from mudata import MuData
        mdata = MuData({feature_key: input, "milo": AnnData()})
        return mdata

    def make_nhoods(
        self,
        data: AnnData,
        neighbors_key: str | None = None,
        feature_key: str | None = "rna",
        prop: float = 0.1,
        seed: int = 0,
        copy: bool = False,
    ):
        """Randomly sample vertices on a KNN graph to define neighbourhoods of cells.

        The set of neighborhoods get refined by computing the median profile for the neighbourhood in reduced dimensional space
        and by selecting the nearest vertex to this position.
        Thus, multiple neighbourhoods may be collapsed to prevent over-sampling the graph space.

        Args:
            data: AnnData object with KNN graph defined in `obsp` or MuData object with a modality with KNN graph defined in `obsp`
            neighbors_key: The key in `adata.obsp` or `mdata[feature_key].obsp` to use as KNN graph.
                           If not specified, `make_nhoods` looks .obsp['connectivities'] for connectivities (default storage places for `scanpy.pp.neighbors`).
                           If specified, it looks at .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
            feature_key: If input data is MuData, specify key to cell-level AnnData object.
            prop: Fraction of cells to sample for neighbourhood index search.
            seed: Random seed for cell sampling.
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            If `copy=True`, returns the copy of `adata` with the result in `.obs`, `.obsm`, and `.uns`.
            Otherwise:

            nhoods: scipy.sparse._csr.csr_matrix in `adata.obsm['nhoods']`.
            A binary matrix of cell to neighbourhood assignments. Neighbourhoods in the columns are ordered by the order of the index cell in adata.obs_names

            nhood_ixs_refined: pandas.Series in `adata.obs['nhood_ixs_refined']`.
            A boolean indicating whether a cell is an index for a neighbourhood

            nhood_kth_distance: pandas.Series in `adata.obs['nhood_kth_distance']`.
            The distance to the kth nearest neighbour for each index cell (used for SpatialFDR correction)

            nhood_neighbors_key: `adata.uns["nhood_neighbors_key"]`
            KNN graph key, used for neighbourhood construction

        Examples:
            >>> import omicverse as ov
            >>> adata = ov.datasets.bhattacherjee()
            >>> milo = ov.single.Milo()
            >>> mdata = milo.load(adata)
            >>> ov.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])

        """
        from mudata import MuData
        if isinstance(data, MuData):
            adata = data[feature_key]
        if isinstance(data, AnnData):
            adata = data
        if copy:
            adata = adata.copy()

        # Get reduced dim used for KNN graph
        if neighbors_key is None:
            try:
                use_rep = adata.uns["neighbors"]["params"]["use_rep"]
            except KeyError:
                _log_warning("Using X_pca as default embedding")
                use_rep = "X_pca"
            try:
                knn_graph = adata.obsp["connectivities"].copy()
            except KeyError:
                _log_error('No "connectivities" slot in adata.obsp -- please run scanpy.pp.neighbors(adata) first')
                raise
        else:
            try:
                use_rep = adata.uns[neighbors_key]["params"]["use_rep"]
            except KeyError:
                _log_warning("Using X_pca as default embedding")
                use_rep = "X_pca"
            knn_graph = adata.obsp[neighbors_key + "_connectivities"].copy()

        X_dimred = adata.obsm[use_rep]
        n_ixs = int(np.round(adata.n_obs * prop))
        knn_graph[knn_graph != 0] = 1
        random.seed(seed)
        random_vertices = random.sample(range(adata.n_obs), k=n_ixs)
        random_vertices.sort()
        ixs_nn = knn_graph[random_vertices, :]
        non_zero_rows = ixs_nn.nonzero()[0]
        non_zero_cols = ixs_nn.nonzero()[1]
        refined_vertices = np.empty(
            shape=[
                len(random_vertices),
            ]
        )

        for i in range(len(random_vertices)):
            nh_pos = np.median(X_dimred[non_zero_cols[non_zero_rows == i], :], 0).reshape(-1, 1)
            nn_ixs = non_zero_cols[non_zero_rows == i]
            # Find closest real point (amongst nearest neighbors)
            dists = euclidean_distances(X_dimred[non_zero_cols[non_zero_rows == i], :], nh_pos.T)
            # Update vertex index
            refined_vertices[i] = nn_ixs[dists.argmin()]

        refined_vertices = np.unique(refined_vertices.astype("int"))
        refined_vertices.sort()

        nhoods = knn_graph[:, refined_vertices]
        adata.obsm["nhoods"] = nhoods

        # Add ixs to adata
        adata.obs["nhood_ixs_random"] = adata.obs_names.isin(adata.obs_names[random_vertices])
        adata.obs["nhood_ixs_refined"] = adata.obs_names.isin(adata.obs_names[refined_vertices])
        adata.obs["nhood_ixs_refined"] = adata.obs["nhood_ixs_refined"].astype("int")
        adata.obs["nhood_ixs_random"] = adata.obs["nhood_ixs_random"].astype("int")
        adata.uns["nhood_neighbors_key"] = neighbors_key
        # Store distance to K-th nearest neighbor (used for spatial FDR correction)
        knn_dists = adata.obsp["distances"] if neighbors_key is None else adata.obsp[neighbors_key + "_distances"]

        nhood_ixs = adata.obs["nhood_ixs_refined"] == 1
        dist_mat = knn_dists[np.asarray(nhood_ixs), :]
        k_distances = dist_mat.max(1).toarray().ravel()
        adata.obs["nhood_kth_distance"] = 0
        adata.obs["nhood_kth_distance"] = adata.obs["nhood_kth_distance"].astype(float)
        adata.obs.loc[adata.obs["nhood_ixs_refined"] == 1, "nhood_kth_distance"] = k_distances

        if copy:
            return adata

    def count_nhoods(
        self,
        data: AnnData,
        sample_col: str,
        feature_key: str | None = "rna",
    ):
        """Builds a sample-level AnnData object storing the matrix of cell counts per sample per neighbourhood.

        Args:
            data: AnnData object with neighbourhoods defined in `obsm['nhoods']` or MuData object with a modality with neighbourhoods defined in `obsm['nhoods']`
            sample_col: Column in adata.obs that contains sample information
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            MuData object storing the original (i.e. rna) AnnData in `mudata[feature_key]`
            and the compositional anndata storing the neighbourhood cell counts in `mudata['milo']`.
            Here:
            - `mudata['milo'].obs_names` are samples (defined from `adata.obs['sample_col']`)
            - `mudata['milo'].var_names` are neighbourhoods
            - `mudata['milo'].X` is the matrix counting the number of cells from each
            sample in each neighbourhood

        Examples:
            >>> import omicverse as ov
            >>> adata = ov.datasets.bhattacherjee()
            >>> milo = ov.single.Milo()
            >>> mdata = milo.load(adata)
            >>> ov.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")

        """
        from mudata import MuData
        if isinstance(data, MuData):
            adata = data[feature_key]
            is_MuData = True
        if isinstance(data, AnnData):
            adata = data
            is_MuData = False
        if isinstance(adata, AnnData):
            try:
                nhoods = adata.obsm["nhoods"]
            except KeyError:
                _log_error('Cannot find "nhoods" slot in adata.obsm -- please run milopy.make_nhoods(adata)')
                raise
        # Make nhood abundance matrix
        sample_dummies = pd.get_dummies(adata.obs[sample_col])
        all_samples = sample_dummies.columns
        sample_dummies = csr_matrix(sample_dummies.values)
        nhood_count_mat = nhoods.T.dot(sample_dummies)
        sample_obs = pd.DataFrame(index=all_samples)
        sample_adata = AnnData(X=nhood_count_mat.T, obs=sample_obs)
        sample_adata.uns["sample_col"] = sample_col
        # Save nhood index info
        sample_adata.var["index_cell"] = adata.obs_names[adata.obs["nhood_ixs_refined"] == 1]
        sample_adata.var["kth_distance"] = adata.obs.loc[
            adata.obs["nhood_ixs_refined"] == 1, "nhood_kth_distance"
        ].values

        if is_MuData is True:
            data.mod["milo"] = sample_adata
            return data
        else:
            milo_mdata = MuData({feature_key: adata, "milo": sample_adata})
            return milo_mdata

    def da_nhoods(
        self,
        mdata,
        design: str,
        model_contrasts: str | None = None,
        subset_samples: list[str] | None = None,
        add_intercept: bool = True,
        feature_key: str | None = "rna",
        solver: Literal["edger", "batchglm"] = "edger",
    ):
        """Performs differential abundance testing on neighbourhoods using QLF test implementation.

        Args:
            mdata: MuData object
            design: Formula for the test, following glm syntax from R (e.g. '~ condition').
                    Terms should be columns in `milo_mdata[feature_key].obs`.
            model_contrasts: A string vector that defines the contrasts used to perform DA testing, following glm syntax from R (e.g. "conditionDisease - conditionControl").
                             If no contrast is specified (default), then the last categorical level in condition of interest is used as the test group.
            subset_samples: subset of samples (obs in `milo_mdata['milo']`) to use for the test.
            add_intercept: whether to include an intercept in the model. If False, this is equivalent to adding + 0 in the design formula. When model_contrasts is specified, this is set to False by default.
            feature_key: If input data is MuData, specify key to cell-level AnnData object.
            solver: The solver to fit the model to. One of "edger" or "batchglm" (currently only "edger" is implemented)

        Returns:
            None, modifies `milo_mdata['milo']` in place, adding the results of the DA test to `.var`:
            - `logFC` stores the log fold change in cell abundance (coefficient from the GLM)
            - `PValue` stores the p-value for the QLF test before multiple testing correction
            - `SpatialFDR` stores the p-value adjusted for multiple testing to limit the false discovery rate,
                calculated with weighted Benjamini-Hochberg procedure

        Examples:
            >>> import omicverse as ov
            >>> adata = ov.datasets.bhattacherjee()
            >>> milo = ov.single.Milo()
            >>> mdata = milo.load(adata)
            >>> ov.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata, design="~label")

        """
        import patsy
        from ..external import pyedger as _edger
        try:
            sample_adata = mdata["milo"]
        except KeyError:
            _log_error(
                "milo_mdata should be a MuData object with two slots:"
                " feature_key and 'milo' - please run milopy.count_nhoods() first"
            )
            raise
        adata = mdata[feature_key]

        covariates = [x.strip(" ") for x in set(re.split("\\+|\\*", design.lstrip("~ ")))]

        # Add covariates used for testing to sample_adata.var
        sample_col = sample_adata.uns["sample_col"]
        try:
            sample_obs = adata.obs[covariates + [sample_col]].drop_duplicates()
        except KeyError:
            missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
            _log_warning("Covariates {c} are not columns in adata.obs".format(c=" ".join(missing_cov)))
            raise
        sample_obs = sample_obs[covariates + [sample_col]]
        sample_obs.index = sample_obs[sample_col].astype("str")

        try:
            assert sample_obs.loc[sample_adata.obs_names].shape[0] == len(sample_adata.obs_names)
        except AssertionError:
            _log_warning(
                f"Values in mdata[{feature_key}].obs[{covariates}] cannot be unambiguously assigned to each sample"
                f" -- each sample value should match a single covariate value"
            )
            raise
        sample_adata.obs = sample_obs.loc[sample_adata.obs_names]

        # Get design dataframe
        try:
            design_df = sample_adata.obs[covariates]
        except KeyError:
            missing_cov = [x for x in covariates if x not in sample_adata.obs.columns]
            _log_error(
                'Covariates {c} are not columns in adata.uns["sample_adata"].obs'.format(c=" ".join(missing_cov))
            )
            raise
        # Get count matrix
        count_mat = sample_adata.X.T.toarray()
        lib_size = count_mat.sum(0)

        # Filter out samples with zero counts
        keep_smp = lib_size > 0

        # Subset samples
        if subset_samples is not None:
            keep_smp = keep_smp & sample_adata.obs_names.isin(subset_samples)
            design_df = design_df[keep_smp]
            for i, e in enumerate(design_df.columns):
                if design_df.dtypes[i].name == "category":
                    design_df[e] = design_df[e].cat.remove_unused_categories()

        # Filter out nhoods with zero counts (they can appear after sample filtering)
        keep_nhoods = count_mat[:, keep_smp].sum(1) > 0

        if solver == "edger":
            # Define model matrix using patsy
            if not add_intercept or model_contrasts is not None:
                design_formula = design + " + 0"
            else:
                design_formula = design

            # Create model matrix
            # Keep as patsy DesignMatrix (don't convert to array)
            model_matrix = patsy.dmatrix(design_formula, design_df)

            # Design matrix as a plain ndarray for pyedger.
            X = np.asarray(model_matrix, dtype=float)
            Y = count_mat[keep_nhoods, :][:, keep_smp]

            # edgeR TMM normalization factors. pyedger's DGEList applies the
            # effective library size (lib.size * norm.factors) internally — the
            # same way Bioconductor edgeR / miloR do — so we pass the TMM
            # factors as norm_factors rather than pre-multiplying lib sizes.
            _log_info("Calculating TMM normalization factors...")
            tmm_factors = calcNormFactors(Y, method="TMM")
            _log_info(f"TMM factors: {tmm_factors}")
            # ``group`` is only required to satisfy pyedger's API; with an
            # explicit ``design`` the dispersion is estimated from the design
            # matrix (GLM path), not the group, so a placeholder is fine.
            dge = _edger.DGEList(
                counts=Y,
                norm_factors=np.asarray(tmm_factors, dtype=float),
                group=np.zeros(Y.shape[1]),
            )

            # Dispersion + quasi-likelihood GLM against the design — exactly the
            # engine miloR::testNhoods wraps (edgeR estimateDisp -> glmQLFit ->
            # glmQLFTest), via the pure-Python pyedger port. No inmoose / R.
            # Note: pyedger's design-based dispersion uses trend_method='none'
            # and robust QL moderation is not yet implemented (robust=False),
            # which also matches the previous inmoose behaviour. R-parity vs
            # Bioconductor edgeR (same options) is pinned in
            # tests/single/test_milo_pyedger_parity.py.
            _edger.estimateDisp(dge, design=X, trend_method="none", robust=False)
            # legacy=False is edgeR 4.x's (and miloR's) default bias-adjusted QL
            # workflow; it tracks Bioconductor edgeR's F/PValue much more
            # closely than the classic legacy=True path.
            fit = _edger.glmQLFit(dge, design=X, robust=False, legacy=False)

            # Test
            if model_contrasts is not None:
                # Parse contrasts and create contrast vector
                # Get column names from patsy DesignMatrix
                contrast_cols = list(model_matrix.design_info.column_names)
                _log_info(f"Design matrix columns: {contrast_cols}")
                _log_info(f"Model contrasts: {model_contrasts}")

                # Parse the contrast string (e.g., "condition[Salmonella]-condition[Control]")
                contrast_terms = re.split(r'\s*[+-]\s*', model_contrasts.strip())
                signs = [1] + [-1 if s == '-' else 1 for s in re.findall(r'[+-]', model_contrasts)]

                # Create contrast vector
                # CRITICAL: When design has no intercept (+ 0), we MUST use contrast vector
                # Testing a single coefficient would give the mean in that group, not logFC
                contrast_vector = np.zeros(len(contrast_cols))

                for term, sign in zip(contrast_terms, signs):
                    term = term.strip()
                    # Try exact match first
                    if term in contrast_cols:
                        idx = contrast_cols.index(term)
                        contrast_vector[idx] = sign
                        _log_info(f"  Contrast term '{term}' (index {idx}): {sign:+d}")
                    else:
                        # Try with T. prefix (patsy treatment coding with intercept)
                        # condition[Treatment] -> condition[T.Treatment]
                        if '[' in term and ']' in term:
                            base, level = term.split('[')
                            level = level.rstrip(']')
                            term_with_t = f"{base}[T.{level}]"
                            if term_with_t in contrast_cols:
                                idx = contrast_cols.index(term_with_t)
                                contrast_vector[idx] = sign
                                _log_info(f"  Contrast term '{term_with_t}' (index {idx}): {sign:+d}")
                            else:
                                _log_error(f"Could not find contrast term '{term}' in columns: {contrast_cols}")
                                raise ValueError(f"Invalid contrast term: {term}")
                        else:
                            _log_error(f"Could not find contrast term '{term}' in columns: {contrast_cols}")
                            raise ValueError(f"Invalid contrast term: {term}")

                # Verify contrast vector
                if contrast_vector.sum() == 0 and np.abs(contrast_vector).sum() > 0:
                    _log_info(f"Contrast vector: {contrast_vector}")
                    _log_info("✓ Contrast sums to 0 (valid difference contrast)")
                else:
                    _log_warning(f"Contrast vector: {contrast_vector}")
                    _log_warning("⚠ Contrast does not sum to 0")

                # Perform test with contrast — tests the difference
                # test_group − control_group (pyedger accepts a 1-D contrast).
                res_obj = _edger.glmQLFTest(fit, contrast=contrast_vector)
            else:
                # Test the last coefficient (0-based index).
                n_coef = model_matrix.shape[1]
                res_obj = _edger.glmQLFTest(fit, coef=n_coef - 1)

            # pyedger returns a result object whose ``.table`` is in nhood order
            # with edgeR column names (logFC, logCPM, F, PValue). Add the
            # non-spatial BH FDR (Milo's graph SpatialFDR is computed below).
            res = res_obj.table.copy()
            from statsmodels.stats.multitest import multipletests
            _pv = res["PValue"].to_numpy(dtype=float)
            res["FDR"] = multipletests(np.nan_to_num(_pv, nan=1.0),
                                       method="fdr_bh")[1]

        elif solver == "batchglm":
            raise NotImplementedError("batchglm solver is not yet implemented")

        # Save outputs
        res.index = sample_adata.var_names[keep_nhoods]

        # Debug: print column names
        _log_info(f"edgeR (pyedger) result columns: {list(res.columns)}")

        if any(col in sample_adata.var.columns for col in res.columns):
            sample_adata.var = sample_adata.var.drop(res.columns, axis=1)
        sample_adata.var = pd.concat([sample_adata.var, res], axis=1)

        # Run Graph spatial FDR correction
        self._graph_spatial_fdr(sample_adata, neighbors_key=adata.uns["nhood_neighbors_key"])

    def annotate_nhoods(
        self,
        mdata,
        anno_col: str,
        feature_key: str | None = "rna",
    ):
        """Assigns a categorical label to neighbourhoods, based on the most frequent label among cells in each neighbourhood. This can be useful to stratify DA testing results by cell types or samples.

        Args:
            mdata: MuData object
            anno_col: Column in adata.obs containing the cell annotations to use for nhood labelling
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            Adds in place.
            - `milo_mdata['milo'].var["nhood_annotation"]`: assigning a label to each nhood
            - `milo_mdata['milo'].var["nhood_annotation_frac"]` stores the fraciton of cells in the neighbourhood with the assigned label
            - `milo_mdata['milo'].varm['frac_annotation']`: stores the fraction of cells from each label in each nhood
            - `milo_mdata['milo'].uns["annotation_labels"]`: stores the column names for `milo_mdata['milo'].varm['frac_annotation']`

        Examples:
            >>> import omicverse as ov
            >>> adata = ov.datasets.bhattacherjee()
            >>> milo = ov.single.Milo()
            >>> mdata = milo.load(adata)
            >>> ov.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.annotate_nhoods(mdata, anno_col="cell_type")

        """
        try:
            sample_adata = mdata["milo"]
        except KeyError:
            _log_error(
                "milo_mdata should be a MuData object with two slots: feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]

        # Check value is not numeric
        if pd.api.types.is_numeric_dtype(adata.obs[anno_col]):
            raise ValueError(
                "adata.obs[anno_col] is not of categorical type - please use milopy.utils.annotate_nhoods_continuous for continuous variables"
            )

        anno_dummies = pd.get_dummies(adata.obs[anno_col])
        anno_count = adata.obsm["nhoods"].T.dot(csr_matrix(anno_dummies.values))
        anno_count_dense = anno_count.toarray()
        anno_sum = anno_count_dense.sum(1)
        anno_frac = np.divide(anno_count_dense, anno_sum[:, np.newaxis])

        anno_frac_dataframe = pd.DataFrame(anno_frac, columns=anno_dummies.columns, index=sample_adata.var_names)
        sample_adata.varm["frac_annotation"] = anno_frac_dataframe.values
        sample_adata.uns["annotation_labels"] = anno_frac_dataframe.columns
        sample_adata.uns["annotation_obs"] = anno_col
        sample_adata.var["nhood_annotation"] = anno_frac_dataframe.idxmax(1)
        sample_adata.var["nhood_annotation_frac"] = anno_frac_dataframe.max(1)

    def annotate_nhoods_continuous(self, mdata, anno_col: str, feature_key: str | None = "rna"):
        """Assigns a continuous value to neighbourhoods, based on mean cell level covariate stored in adata.obs. This can be useful to correlate DA log-foldChanges with continuous covariates such as pseudotime, gene expression scores etc...

        Args:
            mdata: MuData object
            anno_col: Column in adata.obs containing the cell annotations to use for nhood labelling
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            Adds in place.
            - `milo_mdata['milo'].var["nhood_{anno_col}"]`: assigning a continuous value to each nhood

        Examples:
            >>> import omicverse as ov
            >>> adata = ov.datasets.bhattacherjee()
            >>> milo = ov.single.Milo()
            >>> mdata = milo.load(adata)
            >>> ov.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.annotate_nhoods_continuous(mdata, anno_col="nUMI")
        """
        if "milo" not in mdata.mod:
            raise ValueError(
                "milo_mdata should be a MuData object with two slots: feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
            )
        adata = mdata[feature_key]

        # Check value is not categorical
        if not pd.api.types.is_numeric_dtype(adata.obs[anno_col]):
            raise ValueError(
                "adata.obs[anno_col] is not of continuous type - please use milopy.utils.annotate_nhoods for categorical variables"
            )

        anno_val = adata.obsm["nhoods"].T.dot(csr_matrix(adata.obs[anno_col]).T)

        mean_anno_val = anno_val.toarray() / np.array(adata.obsm["nhoods"].T.sum(1))

        mdata["milo"].var[f"nhood_{anno_col}"] = mean_anno_val

    def add_covariate_to_nhoods_var(self, mdata, new_covariates: list[str], feature_key: str | None = "rna"):
        """Add covariate from cell-level obs to sample-level obs. These should be covariates for which a single value can be assigned to each sample.

        Args:
            mdata: MuData object
            new_covariates: columns in `milo_mdata[feature_key].obs` to add to `milo_mdata['milo'].obs`.
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            None, adds columns to `milo_mdata['milo']` in place

        Examples:
            >>> import omicverse as ov
            >>> adata = ov.datasets.bhattacherjee()
            >>> milo = ov.single.Milo()
            >>> mdata = milo.load(adata)
            >>> ov.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.add_covariate_to_nhoods_var(mdata, new_covariates=["label"])
        """
        try:
            sample_adata = mdata["milo"]
        except KeyError:
            _log_error(
                "milo_mdata should be a MuData object with two slots: feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]

        sample_col = sample_adata.uns["sample_col"]
        covariates = list(
            set(sample_adata.obs.columns[sample_adata.obs.columns != sample_col].tolist() + new_covariates)
        )
        try:
            sample_obs = adata.obs[covariates + [sample_col]].drop_duplicates()
        except KeyError:
            missing_cov = [covar for covar in covariates if covar not in sample_adata.obs.columns]
            _log_error("Covariates {c} are not columns in adata.obs".format(c=" ".join(missing_cov)))
            raise
        sample_obs = sample_obs[covariates + [sample_col]].astype("str")
        sample_obs.index = sample_obs[sample_col]
        try:
            assert sample_obs.loc[sample_adata.obs_names].shape[0] == len(sample_adata.obs_names)
        except ValueError:
            _log_error(
                "Covariates cannot be unambiguously assigned to each sample -- each sample value should match a single covariate value"
            )
            raise
        sample_adata.obs = sample_obs.loc[sample_adata.obs_names]

    def build_nhood_graph(self, mdata, basis: str = "X_umap", feature_key: str | None = "rna"):
        """Build graph of neighbourhoods used for visualization of DA results.

        Args:
            mdata: MuData object
            basis: Name of the obsm basis to use for layout of neighbourhoods (key in `adata.obsm`).
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            - `milo_mdata['milo'].varp['nhood_connectivities']`: graph of overlap between neighbourhoods (i.e. no of shared cells)
            - `milo_mdata['milo'].var["Nhood_size"]`: number of cells in neighbourhoods

        Examples:
            >>> import omicverse as ov
            >>> import scanpy as sc
            >>> adata = ov.datasets.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> ov.pp.neighbors(mdata["rna"])
            >>> ov.pp.umap(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.build_nhood_graph(mdata)
        """
        adata = mdata[feature_key]
        # # Add embedding positions
        mdata["milo"].varm["X_milo_graph"] = adata[adata.obs["nhood_ixs_refined"] == 1].obsm[basis]
        # Add nhood size
        mdata["milo"].var["Nhood_size"] = np.array(adata.obsm["nhoods"].sum(0)).flatten()
        # Add adjacency graph
        mdata["milo"].varp["nhood_connectivities"] = adata.obsm["nhoods"].T.dot(adata.obsm["nhoods"])
        mdata["milo"].varp["nhood_connectivities"].setdiag(0)
        mdata["milo"].varp["nhood_connectivities"].eliminate_zeros()
        mdata["milo"].uns["nhood"] = {
            "connectivities_key": "nhood_connectivities",
            "distances_key": "",
        }

    def add_nhood_expression(self, mdata, layer: str | None = None, feature_key: str | None = "rna") -> None:
        """Calculates the mean expression in neighbourhoods of each feature.

        Args:
            mdata: MuData object
            layer: If provided, use `milo_mdata[feature_key][layer]` as expression matrix instead of `milo_mdata[feature_key].X`.
            feature_key: If input data is MuData, specify key to cell-level AnnData object.

        Returns:
            Updates adata in place to store the matrix of average expression in each neighbourhood in `milo_mdata['milo'].varm['expr']`

        Examples:
            >>> import omicverse as ov
            >>> adata = ov.datasets.bhattacherjee()
            >>> milo = ov.single.Milo()
            >>> mdata = milo.load(adata)
            >>> ov.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.add_nhood_expression(mdata)

        """
        try:
            sample_adata = mdata["milo"]
        except KeyError:
            _log_error(
                "milo_mdata should be a MuData object with two slots:"
                " feature_key and 'milo' - please run milopy.count_nhoods(adata) first"
            )
            raise
        adata = mdata[feature_key]

        # Get gene expression matrix
        if layer is None:
            X = adata.X
            expr_id = "expr"
        else:
            X = adata.layers[layer]
            expr_id = "expr_" + layer

        # Aggregate over nhoods -- taking the mean
        nhoods_X = X.T.dot(adata.obsm["nhoods"])
        nhoods_X = csr_matrix(nhoods_X / adata.obsm["nhoods"].toarray().sum(0))
        sample_adata.varm[expr_id] = nhoods_X.T

    def _graph_spatial_fdr(
        self,
        sample_adata: AnnData,
        neighbors_key: str | None = None,
    ):
        """FDR correction weighted on inverse of connectivity of neighbourhoods.

        The distance to the k-th nearest neighbor is used as a measure of connectivity.

        Args:
            sample_adata: Sample-level AnnData.
            neighbors_key: The key in `adata.obsp` to use as KNN graph.
        """
        # use 1/connectivity as the weighting for the weighted BH adjustment from Cydar
        w = 1 / sample_adata.var["kth_distance"]
        w[np.isinf(w)] = 0

        # Computing a density-weighted q-value.
        pvalues = sample_adata.var["PValue"]
        keep_nhoods = ~pvalues.isna()  # Filtering in case of test on subset of nhoods
        o = pvalues[keep_nhoods].argsort()
        pvalues = pvalues.loc[keep_nhoods].iloc[o]
        w = w.loc[keep_nhoods].iloc[o]

        adjp = np.zeros(shape=len(o))
        adjp[o] = (sum(w) * pvalues / np.cumsum(w))[::-1].cummin()[::-1]
        adjp = np.array([x if x < 1 else 1 for x in adjp])

        sample_adata.var["SpatialFDR"] = np.nan
        sample_adata.var.loc[keep_nhoods, "SpatialFDR"] = adjp

    # Removed _setup_rpy2 and _try_import_bioc_library methods as they are no longer needed

    # The plotting methods remain unchanged
    #@_doc_params(common_plot_args=doc_common_plot_args)
    def plot_nhood_graph(  # pragma: no cover # noqa: D417
        self,
        mdata,
        *,
        alpha: float = 0.1,
        min_logFC: float = 0,
        min_size: int = 10,
        plot_edges: bool = False,
        title: str = "DA log-Fold Change",
        color_map: Colormap | str | None = None,
        palette: str | Sequence[str] | None = None,
        ax: Axes | None = None,
        return_fig: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Visualize DA results on abstracted graph (wrapper around ov.pl.embedding).

        Args:
            mdata: MuData object
            alpha: Significance threshold. (default: 0.1)
            min_logFC: Minimum absolute log-Fold Change to show results. If is 0, show all significant neighbourhoods.
            min_size: Minimum size of nodes in visualization. (default: 10)
            plot_edges: If edges for neighbourhood overlaps whould be plotted.
            title: Plot title.
            {common_plot_args}
            **kwargs: Additional arguments to `ov.pl.embedding`.

        Examples:
            >>> import omicverse as ov
            >>> adata = ov.datasets.bhattacherjee()
            >>> milo = ov.single.Milo()
            >>> mdata = milo.load(adata)
            >>> ov.pp.neighbors(mdata["rna"])
            >>> ov.pp.umap(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata,
            >>>            design='~label',
            >>>            model_contrasts='labelwithdraw_15d_Cocaine-labelwithdraw_48h_Cocaine')
            >>> milo.build_nhood_graph(mdata)
            >>> milo.plot_nhood_graph(mdata)

        Preview:
            .. image:: /_static/docstring_previews/milo_nhood_graph.png
        """
        from ..pl._single import embedding
        nhood_adata = mdata["milo"].T.copy()

        if "Nhood_size" not in nhood_adata.obs.columns:
            raise KeyError(
                'Cannot find "Nhood_size" column in adata.uns["nhood_adata"].obs -- \
                    please run milopy.utils.build_nhood_graph(adata)'
            )

        nhood_adata.obs["graph_color"] = nhood_adata.obs["logFC"]
        nhood_adata.obs.loc[nhood_adata.obs["SpatialFDR"] > alpha, "graph_color"] = np.nan
        nhood_adata.obs["abs_logFC"] = abs(nhood_adata.obs["logFC"])
        nhood_adata.obs.loc[nhood_adata.obs["abs_logFC"] < min_logFC, "graph_color"] = np.nan

        # Plotting order - extreme logFC on top
        nhood_adata.obs.loc[nhood_adata.obs["graph_color"].isna(), "abs_logFC"] = np.nan
        ordered = nhood_adata.obs.sort_values("abs_logFC", na_position="first").index
        nhood_adata = nhood_adata[ordered]

        vmax = np.max([nhood_adata.obs["graph_color"].max(), abs(nhood_adata.obs["graph_color"].min())])
        vmin = -vmax

        fig = embedding(
            nhood_adata,
            "X_milo_graph",
            color="graph_color",
            cmap="RdBu_r",
            size=nhood_adata.obs["Nhood_size"] * min_size,
            edges=plot_edges,
            neighbors_key="nhood",
            sort_order=False,
            frameon=False,
            vmax=vmax,
            vmin=vmin,
            title=title,
            color_map=color_map,
            palette=palette,
            ax=ax,
            show=False,
            **kwargs,
        )

        if return_fig:
            return fig
        plt.show()
        return None

    #@_doc_params(common_plot_args=doc_common_plot_args)
    def plot_nhood(  # pragma: no cover # noqa: D417
        self,
        mdata,
        ix: int,
        *,
        feature_key: str | None = "rna",
        basis: str = "X_umap",
        color_map: Colormap | str | None = None,
        palette: str | Sequence[str] | None = None,
        ax: Axes | None = None,
        return_fig: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Visualize cells in a neighbourhood.

        Args:
            mdata: MuData object with feature_key slot, storing neighbourhood assignments in `mdata[feature_key].obsm['nhoods']`
            ix: index of neighbourhood to visualize
            feature_key: Key in mdata to the cell-level AnnData object.
            basis: Embedding to use for visualization.
            color_map: Colormap to use for coloring.
            palette: Color palette to use for coloring.
            ax: Axes to plot on.
            {common_plot_args}
            **kwargs: Additional arguments to `scanpy.pl.embedding`.

        Examples:
            >>> import omicverse as ov
            >>> import scanpy as sc
            >>> adata = ov.datasets.bhattacherjee()
            >>> milo = ov.single.Milo()
            >>> mdata = milo.load(adata)
            >>> ov.pp.neighbors(mdata["rna"])
            >>> ov.pp.umap(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> milo.plot_nhood(mdata, ix=0)

        Preview:
            .. image:: /_static/docstring_previews/milo_nhood.png
        """
        from ..pl._single import embedding
        mdata[feature_key].obs["Nhood"] = mdata[feature_key].obsm["nhoods"][:, ix].toarray().ravel()
        fig = embedding(
            mdata[feature_key],
            basis,
            color="Nhood",
            size=30,
            title="Nhood" + str(ix),
            color_map=color_map,
            palette=palette,
            return_fig=return_fig,
            ax=ax,
            show=False,
            **kwargs,
        )

        if return_fig:
            return fig
        plt.show()
        return None

    #@_doc_params(common_plot_args=doc_common_plot_args)
    def plot_da_beeswarm(  # pragma: no cover # noqa: D417
        self,
        mdata,
        *,
        feature_key: str | None = "rna",
        anno_col: str = "nhood_annotation",
        alpha: float = 0.1,
        subset_nhoods: list[str] = None,
        palette: str | Sequence[str] | dict[str, str] | None = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plot beeswarm plot of logFC against nhood labels.

        Args:
            mdata: MuData object
            feature_key: Key in mdata to the cell-level AnnData object.
            anno_col: Column in adata.uns['nhood_adata'].obs to use as annotation. (default: 'nhood_annotation'.)
            alpha: Significance threshold. (default: 0.1)
            subset_nhoods: List of nhoods to plot. If None, plot all nhoods.
            palette: Name of Seaborn color palette for violinplots.
                     Defaults to pre-defined category colors for violinplots.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import omicverse as ov
            >>> adata = ov.datasets.bhattacherjee()
            >>> milo = ov.single.Milo()
            >>> mdata = milo.load(adata)
            >>> ov.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata, design="~label")
            >>> milo.annotate_nhoods(mdata, anno_col="cell_type")
            >>> milo.plot_da_beeswarm(mdata)

        Preview:
            .. image:: /_static/docstring_previews/milo_da_beeswarm.png
        """
        try:
            nhood_adata = mdata["milo"].T.copy()
        except KeyError:
            raise RuntimeError(
                "mdata should be a MuData object with two slots: feature_key and 'milo'. Run 'milopy.count_nhoods(adata)' first."
            ) from None

        try:
            nhood_adata.obs[anno_col]
        except KeyError:
            raise RuntimeError(
                f"Unable to find {anno_col} in mdata['milo'].var. Run 'milopy.utils.annotate_nhoods(adata, anno_col)' first"
            ) from None

        if subset_nhoods is not None:
            nhood_adata = nhood_adata[nhood_adata.obs[anno_col].isin(subset_nhoods)]

        try:
            nhood_adata.obs["logFC"]
        except KeyError:
            raise RuntimeError(
                "Unable to find 'logFC' in mdata.uns['nhood_adata'].obs. Run 'core.da_nhoods(adata)' first."
            ) from None

        sorted_annos = (
            nhood_adata.obs[[anno_col, "logFC"]].groupby(anno_col).median().sort_values("logFC", ascending=True).index
        )

        anno_df = nhood_adata.obs[[anno_col, "logFC", "SpatialFDR"]].copy()
        anno_df["is_signif"] = anno_df["SpatialFDR"] < alpha
        anno_df = anno_df[anno_df[anno_col] != "nan"]

        try:
            obs_col = nhood_adata.uns["annotation_obs"]
            if palette is None:
                palette = dict(
                    zip(
                        mdata[feature_key].obs[obs_col].cat.categories,
                        mdata[feature_key].uns[f"{obs_col}_colors"],
                        strict=False,
                    )
                )
            sns.violinplot(
                data=anno_df,
                y=anno_col,
                x="logFC",
                order=sorted_annos,
                inner=None,
                orient="h",
                palette=palette,
                linewidth=0,
                scale="width",
            )
        except BaseException:  # noqa: BLE001
            sns.violinplot(
                data=anno_df,
                y=anno_col,
                x="logFC",
                order=sorted_annos,
                inner=None,
                orient="h",
                linewidth=0,
                scale="width",
            )
        sns.stripplot(
            data=anno_df,
            y=anno_col,
            x="logFC",
            order=sorted_annos,
            size=2,
            hue="is_signif",
            palette=["grey", "black"],
            orient="h",
            alpha=0.5,
        )
        plt.legend(loc="upper left", title=f"< {int(alpha * 100)}% SpatialFDR", bbox_to_anchor=(1, 1), frameon=False)
        plt.axvline(x=0, ymin=0, ymax=1, color="black", linestyle="--")

        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    #@_doc_params(common_plot_args=doc_common_plot_args)
    def plot_nhood_counts_by_cond(  # pragma: no cover # noqa: D417
        self,
        mdata,
        test_var: str,
        *,
        subset_nhoods: list[str] = None,
        log_counts: bool = False,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plot boxplot of cell numbers vs condition of interest.

        Args:
            mdata: MuData object storing cell level and nhood level information
            test_var: Name of column in adata.obs storing condition of interest (y-axis for boxplot)
            subset_nhoods: List of obs_names for neighbourhoods to include in plot. If None, plot all nhoods.
            log_counts: Whether to plot log1p of cell counts.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.
        """
        try:
            nhood_adata = mdata["milo"].T.copy()
        except KeyError:
            raise RuntimeError(
                "mdata should be a MuData object with two slots: feature_key and 'milo'. Run milopy.count_nhoods(mdata) first"
            ) from None

        if subset_nhoods is None:
            subset_nhoods = nhood_adata.obs_names

        pl_df = pd.DataFrame(nhood_adata[subset_nhoods].X.toarray(), columns=nhood_adata.var_names).melt(
            var_name=nhood_adata.uns["sample_col"], value_name="n_cells"
        )
        pl_df = pd.merge(pl_df, nhood_adata.var)
        pl_df["log_n_cells"] = np.log1p(pl_df["n_cells"])
        if not log_counts:
            sns.boxplot(data=pl_df, x=test_var, y="n_cells", color="lightblue")
            sns.stripplot(data=pl_df, x=test_var, y="n_cells", color="black", s=3)
            plt.ylabel("# cells")
        else:
            sns.boxplot(data=pl_df, x=test_var, y="log_n_cells", color="lightblue")
            sns.stripplot(data=pl_df, x=test_var, y="log_n_cells", color="black", s=3)
            plt.ylabel("log(# cells + 1)")

        plt.xticks(rotation=90)
        plt.xlabel(test_var)

        if return_fig:
            return plt.gcf()
        plt.show()
        return None