r"""
Monocle2-style trajectory analysis for AnnData.

Pure-Python re-implementation of Monocle 2 (Qiu et al. 2017), exposed as
a single `Monocle` class for convenient use within omicverse.

Examples
--------
>>> import omicverse as ov
>>> mono = ov.single.Monocle(adata)
>>> mono.preprocess()
>>> mono.select_ordering_genes()
>>> mono.reduce_dimension(max_components=2)
>>> mono.order_cells()
>>> mono.plot_trajectory(color_by='clusters')
>>> de_results = mono.differential_gene_test()
>>> beam_results = mono.BEAM(branch_point=1)
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from .._registry import register_function

# NOTE: `omicverse.single` is a core domain whose modules must not
# perform `from ..external import ...` at the module top level (see
# ``tests/architecture/test_no_top_level_external_imports.py``). The
# Monocle class below imports the monocle2_py backend once inside
# ``__init__`` and stores it on the instance as ``self._m2``.


def _cell_mst_graph(adata: AnnData):
    """The cell-level weighted MST for whichever reduction was run, or None.

    The two reductions store it in different places, and only one of them
    stores it under ``projected_dp``:

    * **DDRTree** builds the principal graph over Y-CENTRES, then
      ``_project_cells_to_mst`` projects cells onto it and writes the
      cell-level tree to ``projected_dp``.
    * **ICA** has no centres — its MST is already over cells, and
      ``cellPairwiseDistances`` is the cell x cell distance matrix it was
      built from. Nothing ever writes ``projected_dp``.

    Reading ``projected_dp`` alone therefore works on DDRTree and raises on
    ICA, which is what it did. The shape check below is not decoration: on
    DDRTree ``cellPairwiseDistances`` is CENTRES x CENTRES, so using it
    unconditionally would silently index the wrong graph.
    """
    from scipy.sparse import csr_matrix, issparse
    from scipy.sparse.csgraph import minimum_spanning_tree

    monocle = adata.uns.get('monocle', {})
    graph = monocle.get('projected_dp')
    if graph is not None:
        return csr_matrix(graph) if not issparse(graph) else graph.tocsr()

    dp = monocle.get('cellPairwiseDistances')
    if dp is None:
        return None
    dp = np.asarray(dp, dtype=float)
    if dp.shape != (adata.n_obs, adata.n_obs):
        return None                       # centres, not cells — wrong graph
    return minimum_spanning_tree(csr_matrix(dp)).tocsr()


def _population_graph_medoid(adata: AnnData, mask: pd.Series) -> str:
    """Return a matching cell minimizing distance to the target population.

    On a tree, all-node sums of distances to a weighted target set can be
    computed in linear time by rerooting the tree. Restricting the final
    minimum to matching cells gives a deterministic, representative root
    without materializing an all-pairs distance matrix.
    """

    graph = _cell_mst_graph(adata)
    target = np.asarray(mask, dtype=bool)
    target_indices = np.flatnonzero(target)
    if target_indices.size == 0:
        raise ValueError("Cannot select a root medoid without target cells")
    if graph is None:
        raise ValueError(
            "Cannot select a root medoid: no cell-level MST is available for "
            f"dim_reduce_type="
            f"{adata.uns.get('monocle', {}).get('dim_reduce_type')!r}. "
            "Run reduce_dimension() with DDRTree or ICA before order_cells()."
        )

    graph = graph.tocsr().maximum(graph.T.tocsr()).tocsr()
    n_obs = adata.n_obs
    parent = np.full(n_obs, -2, dtype=int)
    edge_to_parent = np.zeros(n_obs, dtype=float)
    distance_from_zero = np.zeros(n_obs, dtype=float)
    parent[0] = -1
    stack = [0]
    traversal = []
    while stack:
        node = stack.pop()
        traversal.append(node)
        for offset in range(graph.indptr[node], graph.indptr[node + 1]):
            neighbor = int(graph.indices[offset])
            if neighbor == parent[node] or parent[neighbor] != -2:
                continue
            parent[neighbor] = node
            weight = float(graph.data[offset])
            edge_to_parent[neighbor] = weight
            distance_from_zero[neighbor] = distance_from_zero[node] + weight
            stack.append(neighbor)
    if len(traversal) != n_obs:
        raise ValueError("Cell projection MST is disconnected")

    subtree_targets = target.astype(int)
    for node in reversed(traversal[1:]):
        subtree_targets[parent[node]] += subtree_targets[node]

    total_targets = int(target_indices.size)
    distance_sum = np.empty(n_obs, dtype=float)
    distance_sum[0] = float(distance_from_zero[target].sum())
    for node in traversal[1:]:
        distance_sum[node] = (
            distance_sum[parent[node]]
            + edge_to_parent[node]
            * (total_targets - 2 * subtree_targets[node])
        )
    root_index = int(
        target_indices[np.argmin(distance_sum[target_indices])]
    )
    return str(adata.obs_names[root_index])


@register_function(
    aliases=["Monocle", "monocle", "monocle2", "monocle 2", "DDRTree trajectory", "BEAM"],
    category="trajectory",
    description=(
        "Monocle 2-style trajectory analysis (pure-Python re-implementation). "
        "Covers size-factor / dispersion estimation, ordering-gene selection, "
        "DDRTree dimension reduction, pseudotime + State assignment, branched "
        "differential expression (BEAM), and the full family of trajectory plots."
    ),
    prerequisites={},
    requires={},
    produces={},
    auto_fix="none",
    examples=[
        "mono = ov.single.Monocle(adata)",
        "mono.preprocess()",
        "mono.select_ordering_genes()",
        "mono.reduce_dimension(max_components=2)",
        "mono.order_cells()",
        "mono.plot_trajectory(color_by='clusters')",
        "beam = mono.BEAM(branch_point=1)",
    ],
    related=["single.pyVIA", "single.dynamic_features", "pl.dynamic_trends"],
)
class Monocle:
    """
    Monocle2-style single-cell trajectory analysis.

    Wraps a pure-Python implementation of Monocle 2 as a stateful analyzer
    operating on an AnnData object. All results are stored in the AnnData
    (``.obs``, ``.var``, ``.uns['monocle']``, ``.obsm``) so the usual scanpy
    workflow continues to work seamlessly.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells × genes). Expression matrix in
        ``adata.X`` should be raw/normalized counts (negative binomial model).

    Attributes
    ----------
    adata : AnnData
        The annotated data matrix with analysis results stored in-place.

    Examples
    --------
    Basic trajectory analysis:

    >>> mono = ov.single.Monocle(adata)
    >>> mono.preprocess()              # size factors + dispersions
    >>> mono.select_ordering_genes()   # high-variance gene selection
    >>> mono.reduce_dimension()        # DDRTree
    >>> mono.order_cells()             # assign pseudotime + State
    >>> mono.plot_trajectory(color_by='clusters')
    >>> mono.plot_genes_in_pseudotime(['Ins1', 'Gcg'])

    Differential expression along pseudotime:

    >>> de = mono.differential_gene_test()
    >>> beam = mono.BEAM(branch_point=1)
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, adata: AnnData):
        """Initialise the Monocle analyser.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix (cells × genes). The expression matrix in
            ``adata.X`` should contain raw or normalised counts — the negative
            binomial model used by size-factor estimation and BEAM assumes
            count-like data. All downstream results (``Pseudotime``, ``State``,
            dispersions, DDRTree reduction) are written back to the same
            AnnData, so subsequent scanpy-style workflows continue to work.

        Notes
        -----
        The external ``monocle2_py`` backend is imported lazily inside this
        constructor (not at module scope) to keep ``omicverse.single`` free of
        top-level ``..external`` imports, per the architecture test.
        """
        from ..external import monocle2_py as _m2  # noqa: PLC0415
        self._m2 = _m2

        self.adata = adata
        self._preprocessed = False
        self._ordering_set = False
        self._reduced = False
        self._ordered = False

    def __repr__(self):
        status = []
        status.append(f"Monocle({self.adata.n_obs} cells × {self.adata.n_vars} genes)")
        if self._preprocessed:
            status.append("  preprocessed: ✓")
        if self._ordering_set:
            n_ord = int(self.adata.var.get('use_for_ordering', pd.Series([False])).sum())
            status.append(f"  ordering genes: {n_ord}")
        if self._reduced:
            method = self.adata.uns.get('monocle', {}).get('dim_reduce_type', 'unknown')
            status.append(f"  reduced: {method}")
        if self._ordered:
            pt = self.adata.obs.get('Pseudotime')
            if pt is not None:
                n_states = self.adata.obs['State'].nunique() if 'State' in self.adata.obs.columns else 0
                status.append(f"  ordered: pseudotime [{pt.min():.2f}, {pt.max():.2f}], "
                              f"{n_states} states")
        return "\n".join(status)

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def detect_genes(self, min_expr: float = 0.1):
        """Flag genes expressed above a threshold.

        Writes ``adata.var['num_cells_expressed']`` and
        ``adata.obs['num_genes_expressed']`` for downstream filtering.

        Parameters
        ----------
        min_expr : float, default 0.1
            Minimum raw count for a cell to count as "expressing" the gene.

        Returns
        -------
        Monocle
            ``self`` for chaining.
        """
        self.adata = self._m2.detect_genes(self.adata, min_expr=min_expr)
        return self

    def estimate_size_factors(self, method: str = 'mean-geometric-mean-total',
                               round_exprs: bool = True):
        """Estimate per-cell size factors.

        Parameters
        ----------
        method : str, default ``'mean-geometric-mean-total'``
            Size-factor estimator. Matches the Monocle 2 R defaults.
        round_exprs : bool, default ``True``
            Round the normalised matrix to integers before fitting dispersions
            — required for the NB model downstream.

        Returns
        -------
        Monocle
            ``self`` for chaining. Writes ``adata.obs['size_factor']``.
        """
        self.adata = self._m2.estimate_size_factors(
            self.adata, method=method, round_exprs=round_exprs,
        )
        return self

    def estimate_dispersions(self, min_cells_detected: int = 1, verbose: bool = False):
        """Fit per-gene dispersions under the negative-binomial model.

        Parameters
        ----------
        min_cells_detected : int, default 1
            Minimum number of expressing cells for a gene to be fit (genes
            below the threshold are dropped from the dispersion table).
        verbose : bool, default ``False``
            Print progress while fitting.

        Returns
        -------
        Monocle
            ``self`` for chaining. Writes empirical / fitted dispersions into
            ``adata.var`` and a dispersion table into ``adata.uns['monocle']``.
        """
        self.adata = self._m2.estimate_dispersions(
            self.adata, min_cells_detected=min_cells_detected, verbose=verbose,
        )
        return self

    def preprocess(self, min_expr: float = 0.1, verbose: bool = False):
        """Run :meth:`detect_genes`, :meth:`estimate_size_factors` and
        :meth:`estimate_dispersions` in sequence.

        Parameters
        ----------
        min_expr : float, default 0.1
            Passed through to :meth:`detect_genes`.
        verbose : bool, default ``False``
            Passed through to :meth:`estimate_dispersions`.

        Returns
        -------
        Monocle
            ``self`` for chaining. Flips ``self._preprocessed = True``.
        """
        self.detect_genes(min_expr=min_expr)
        self.estimate_size_factors()
        self.estimate_dispersions(verbose=verbose)
        self._preprocessed = True
        return self

    def dispersion_table(self) -> pd.DataFrame:
        """Return the per-gene dispersion table populated by
        :meth:`estimate_dispersions`, as a :class:`pandas.DataFrame`.
        """
        return self._m2.dispersion_table(self.adata)

    def relative2abs(self, method: str = 'num_genes',
                     expected_capture_rate: float = 0.25,
                     verbose: bool = False) -> AnnData:
        """Census normalisation — convert TPM / FPKM-scaled counts into
        estimated absolute transcript counts per cell.

        Parameters
        ----------
        method : str, default ``'num_genes'``
            Monocle-style census method.
        expected_capture_rate : float, default 0.25
            Estimated mRNA capture efficiency of the platform.
        verbose : bool, default ``False``
            Print per-cell diagnostics.

        Returns
        -------
        AnnData
            A **new** AnnData with absolute counts in ``.X`` — the original
            ``self.adata`` is left untouched.
        """
        return self._m2.relative2abs(
            self.adata, method=method,
            expected_capture_rate=expected_capture_rate, verbose=verbose,
        )

    # ------------------------------------------------------------------ #
    # Ordering gene selection
    # ------------------------------------------------------------------ #

    def select_ordering_genes(self, genes: Optional[List[str]] = None,
                               mean_expr_thresh: float = 0.1,
                               max_genes: Optional[int] = None):
        """
        Select genes used for trajectory inference.

        Parameters
        ----------
        genes : list of str or None
            If given, use these genes directly. Otherwise auto-select by
            ``dispersion_empirical > dispersion_fit`` and
            ``mean_expression >= mean_expr_thresh``.
        mean_expr_thresh : float
            Minimum mean expression to keep a gene.
        max_genes : int or None
            Cap the number of ordering genes (for large datasets).
        """
        if genes is None:
            disp = self._m2.dispersion_table(self.adata)
            mask = (
                (disp['mean_expression'] >= mean_expr_thresh) &
                (disp['dispersion_empirical'] >= disp['dispersion_fit'])
            )
            genes = disp[mask].index.tolist()

            if max_genes is not None and len(genes) > max_genes:
                ratio = (disp.loc[genes, 'dispersion_empirical']
                         / disp.loc[genes, 'dispersion_fit'])
                genes = ratio.sort_values(ascending=False).head(max_genes).index.tolist()

        self.adata = self._m2.set_ordering_filter(self.adata, genes)
        self._ordering_set = True
        return self

    def set_ordering_filter(self, genes: List[str]):
        """Explicitly set the list of ordering genes."""
        self.adata = self._m2.set_ordering_filter(self.adata, genes)
        self._ordering_set = True
        return self

    # ------------------------------------------------------------------ #
    # Dimension reduction & ordering
    # ------------------------------------------------------------------ #

    def reduce_dimension(self, max_components: int = 2,
                          reduction_method: str = 'DDRTree',
                          norm_method: str = 'log',
                          method: str = 'fast',
                          verbose: bool = False,
                          residualModelFormulaStr: Optional[str] = None,
                          **kwargs):
        """
        Reduce dimensionality and learn the principal graph.

        Parameters
        ----------
        max_components : int
            Number of dimensions to reduce to (2 is standard for visualization).
        reduction_method : {'DDRTree', 'tSNE', 'ICA'}
            Algorithm family to use for the reduction.
        norm_method : {'log', 'none'}
            Gene-expression normalisation applied before the reduction.
        method : {'fast', 'exact'}, default 'fast'
            DDRTree convergence mode (ignored for tSNE/ICA).

            * ``'fast'`` (default) — Reformulated update + sparse
              soft-assignment + cheap stopping criterion
              (``||ΔY||_F / ||Y||_F``).  About 3× faster per call;
              trajectory topology and pseudotime correlation with the
              exact mode are preserved (typically 0.99+), but absolute
              pseudotime values may shift slightly.
            * ``'exact'`` — Matches R Monocle 2 bitwise: evaluates the
              full objective (including the expensive ``||X − WZ||_2^2``
              term) on every iteration and terminates when it stops
              decreasing.  Pass this when you need bitwise R parity.
        residualModelFormulaStr : str or None
            Patsy/R-style model formula specifying cell-level effects to
            subtract before scaling and dimension reduction, for example
            ``"~ batch"`` or ``"~ batch + percent_mito"``. This is the
            pure-Python equivalent of R Monocle 2's parameter of the same
            name. The formula must include an intercept and reference columns
            in ``adata.obs``.
        **kwargs : additional DDRTree parameters
            ``ncenter``, ``lambda_param``, ``param_gamma``, ``sigma``,
            ``maxIter``, ``tol``.
        """
        self.adata = self._m2.reduce_dimension(
            self.adata, max_components=max_components,
            reduction_method=reduction_method,
            norm_method=norm_method, verbose=verbose,
            method=method,
            residualModelFormulaStr=residualModelFormulaStr,
            **kwargs,
        )
        self._reduced = True
        return self

    def order_cells(self, root_state=None, reverse: Optional[bool] = None,
                    root_by_column: Optional[str] = None,
                    root_by_value=None, root_cell: Optional[str] = None):
        """Order cells along the learned trajectory, assigning Pseudotime and State.

        Parameters
        ----------
        root_state : int or None
            Numeric state id to use as the trajectory root. If given,
            the cells in this state are used to locate the root tip on
            the Y-centre MST.
        reverse : bool or None
            If ``True``, flip the default diameter-endpoint choice
            (match R's ``orderCells(reverse=TRUE)``).
        root_by_column : str or None
            Name of a column in ``mono.adata.obs``. If given, the cell
            minimising total tree distance to the matching population — its
            graph medoid — is chosen as an exact root cell.

            This is stricter than R Monocle 2's ``GM_state()`` helper, which
            returns the *State* holding most of the progenitor cells and lets
            ordering pick any cell within it. Anchoring on a specific cell
            stops a State that mixes progenitors with other cells from
            silently rooting the trajectory in the wrong population. The
            calling pattern is unchanged — e.g. for HSMM::

                mono.order_cells()                 # first pass
                mono.order_cells(root_by_column='Hours',
                                 root_by_value=0)  # point root at 0h

            If ``root_by_value`` is ``None``, the minimum value in the
            column is used (so the earliest-observed cells become root).
        root_by_value
            Value within ``root_by_column`` that marks the progenitor
            population. Defaults to the column minimum.
        root_cell : str or None
            Exact observation name to use as the root. When
            ``root_by_column`` is supplied, Monocle chooses the matching
            population's graph medoid and uses it as this exact anchor.
        """
        # Auto-detect root_state from a metadata column if requested.
        # Requires a previous order_cells() call so that `State` exists.
        if root_by_column is not None:
            col = self.adata.obs.get(root_by_column)
            if col is None:
                raise KeyError(
                    f"root_by_column={root_by_column!r} not found in "
                    "adata.obs. Available: "
                    f"{list(self.adata.obs.columns)}"
                )
            if 'State' not in self.adata.obs.columns:
                # Run a first ordering so `State` exists
                self.adata = self._m2.order_cells(self.adata)
                self._ordered = True
            target = root_by_value if root_by_value is not None else col.min()
            mask = col == target
            if not mask.any():
                raise ValueError(
                    f"No cells matched {root_by_column}={target!r}"
                )
            root_cell = _population_graph_medoid(self.adata, mask)
            # An exact population medoid supersedes the State-only heuristic.
            root_state = None

        self.adata = self._m2.order_cells(
            self.adata, root_state=root_state, reverse=reverse,
            root_cell=root_cell,
        )
        if root_by_column is not None:
            self.adata.uns['monocle']['root_by_column'] = root_by_column
            self.adata.uns['monocle']['root_by_value'] = str(target)
            self.adata.uns['monocle']['root_selection_method'] = (
                'target_population_graph_medoid'
            )
        self._ordered = True
        return self

    # ------------------------------------------------------------------ #
    # Clustering
    # ------------------------------------------------------------------ #

    def cluster_cells(self, method: str = 'leiden', k: int = 50,
                      resolution_parameter: float = 0.1, verbose: bool = False,
                      **kwargs):
        """Cluster cells on the reduced-dim space and write labels to
        ``adata.obs['Cluster']``.

        Parameters
        ----------
        method : {'leiden', 'louvain', 'densityPeak', 'DDRTree'}, default 'leiden'
            Clustering algorithm. ``'densityPeak'`` matches the original
            Monocle 2 approach.
        k : int, default 50
            Number of nearest neighbours used to build the k-NN graph
            (ignored by ``'densityPeak'``).
        resolution_parameter : float, default 0.1
            Resolution for Leiden / Louvain — higher values yield more
            clusters.
        verbose : bool, default ``False``
            Print per-step timing.
        **kwargs
            Forwarded to :func:`monocle2_py.cluster_cells`.

        Returns
        -------
        Monocle
            ``self`` for chaining.
        """
        self.adata = self._m2.cluster_cells(
            self.adata, method=method, k=k,
            resolution_parameter=resolution_parameter, verbose=verbose, **kwargs,
        )
        return self

    @staticmethod
    def cluster_genes(expression_matrix, k: int, method: str = 'correlation'):
        """Cluster genes by their expression pattern along pseudotime.

        Unlike the other instance methods, this is a static helper — pass
        the expression matrix (e.g. the output of :meth:`gen_smooth_curves`)
        directly.

        Parameters
        ----------
        expression_matrix : pandas.DataFrame
            Genes (rows) × samples (columns).
        k : int
            Target number of clusters.
        method : {'correlation', 'kmeans', 'hclust'}, default 'correlation'
            Distance metric / clustering backend.

        Returns
        -------
        pandas.Series
            Gene → cluster label mapping.
        """
        from ..external import monocle2_py as _m2  # noqa: PLC0415
        return _m2.cluster_genes(expression_matrix, k, method=method)

    # ------------------------------------------------------------------ #
    # Differential expression
    # ------------------------------------------------------------------ #

    def differential_gene_test(self,
                                fullModelFormulaStr: str = "~sm.ns(Pseudotime, df=3)",
                                reducedModelFormulaStr: str = "~1",
                                relative_expr: bool = True,
                                cores: int = -1, verbose: bool = False) -> pd.DataFrame:
        """Pseudotime-dependent differential expression via a likelihood-ratio
        test between a full GLM (gene ~ f(Pseudotime)) and a reduced null model.

        Parameters
        ----------
        fullModelFormulaStr : str
            Patsy formula for the full model. The default fits a natural
            cubic spline on pseudotime with 3 degrees of freedom.
        reducedModelFormulaStr : str, default ``'~1'``
            Null model (intercept only by default).
        relative_expr : bool, default ``True``
            Normalise by size factors before fitting.
        cores : int, default -1
            Worker processes for the GLM fit. ``-1`` uses all available CPUs.
        verbose : bool, default ``False``
            Print per-gene diagnostics.

        Returns
        -------
        pandas.DataFrame
            One row per gene with columns ``status``, ``pval``, ``qval``,
            ``test_type``. Sorted by ``qval`` so the most significant genes
            come first.
        """
        return self._m2.differential_gene_test(
            self.adata,
            fullModelFormulaStr=fullModelFormulaStr,
            reducedModelFormulaStr=reducedModelFormulaStr,
            relative_expr=relative_expr, cores=cores, verbose=verbose,
        )

    def BEAM(self, branch_point: int = 1, branch_states=None,
              branch_labels=None,
              fullModelFormulaStr: str = "~sm.ns(Pseudotime, df=3)*Branch",
              reducedModelFormulaStr: str = "~sm.ns(Pseudotime, df=3)",
              cores: int = -1, verbose: bool = False) -> pd.DataFrame:
        """Branched Expression Analysis Modelling.

        Tests whether each gene's expression diverges across the two lineages
        downstream of a given branch point. Compares a full model with a
        spline-by-branch interaction against a reduced model that collapses
        the branches into a single spline.

        Parameters
        ----------
        branch_point : int, default 1
            Identifier of the branch point in the learned principal graph
            (1-indexed). See ``mono.branch_points``.
        branch_states : sequence of int, optional
            State IDs assigned to each child lineage. Inferred from the
            graph when ``None``.
        branch_labels : sequence of str, optional
            Human-readable labels for the two child lineages (used in the
            output dataframe / downstream plots).
        fullModelFormulaStr, reducedModelFormulaStr : str
            Patsy formulas for the full and reduced models. Defaults match
            the original Monocle 2 BEAM test.
        cores : int, default -1
            Worker processes for the GLM fit. ``-1`` uses all available CPUs.
        verbose : bool, default ``False``
            Print per-gene diagnostics.

        Returns
        -------
        pandas.DataFrame
            One row per gene with BEAM test statistics, sorted by ``qval``.
            Feed into :meth:`plot_genes_branched_pseudotime` /
            :meth:`plot_genes_branched_heatmap` to visualise the top hits.
        """
        return self._m2.BEAM(
            self.adata, branch_point=branch_point, branch_states=branch_states,
            branch_labels=branch_labels,
            fullModelFormulaStr=fullModelFormulaStr,
            reducedModelFormulaStr=reducedModelFormulaStr,
            cores=cores, verbose=verbose,
        )

    def fit_model(self, modelFormulaStr: str = "~sm.ns(Pseudotime, df=3)",
                   relative_expr: bool = True, cores: int = 1):
        """Fit a per-gene GLM under the NB model.

        Low-level building block used by :meth:`differential_gene_test` and
        :meth:`BEAM`. Most users want one of those higher-level methods.
        Returns the dict of fitted models keyed by gene name.
        """
        return self._m2.fit_model(self.adata, modelFormulaStr=modelFormulaStr,
                              relative_expr=relative_expr, cores=cores)

    def gen_smooth_curves(self, new_data=None,
                           trend_formula: str = "~sm.ns(Pseudotime, df=3)",
                           relative_expr: bool = True, cores: int = 1):
        """Predict smoothed expression trajectories from a fitted model.

        Parameters
        ----------
        new_data : pandas.DataFrame or None
            Design matrix to predict on (e.g. a uniform pseudotime grid).
            When ``None`` the cells' own pseudotime values are used.
        trend_formula : str
            Same formula used when fitting — needed to reconstruct the
            design matrix for ``new_data``.
        relative_expr, cores
            Forwarded to the underlying :func:`fit_model` call.

        Returns
        -------
        pandas.DataFrame
            Genes (rows) × sample points (columns) of smoothed expression.
        """
        return self._m2.gen_smooth_curves(
            self.adata, new_data=new_data, trend_formula=trend_formula,
            relative_expr=relative_expr, cores=cores,
        )

    def cal_ABCs(self, branch_point: int = 1, **kwargs) -> pd.DataFrame:
        """Compute the Area Between Curves for branch-specific genes.

        ABC summarises the divergence of a gene's expression between the two
        lineages after a branch point — higher values mean more branch-
        dependent behaviour. Inputs are routed through :func:`monocle2_py.cal_ABCs`;
        extra kwargs are forwarded verbatim.

        Returns
        -------
        pandas.DataFrame
            One row per gene, columns include ``ABCs`` and supporting stats.
        """
        return self._m2.cal_ABCs(self.adata, branch_point=branch_point, **kwargs)

    def cal_ILRs(self, branch_point: int = 1, return_all: bool = False, **kwargs):
        """Compute the Intrinsic Log-Ratio (per-gene lineage bias).

        Parameters
        ----------
        branch_point : int, default 1
            Branch point identifier (see :attr:`branch_points`).
        return_all : bool, default ``False``
            If ``True``, also return the per-timepoint expression used to
            compute the ratio (useful for plotting).

        Returns
        -------
        pandas.DataFrame or tuple
            ILR dataframe, or (ILR, per-timepoint expression) when
            ``return_all=True``.
        """
        return self._m2.cal_ILRs(
            self.adata, branch_point=branch_point, return_all=return_all, **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    def plot_trajectory(self, color_by: str = 'State', **kwargs):
        """plot_cell_trajectory — main DDRTree trajectory plot."""
        return self._m2.plot_cell_trajectory(self.adata, color_by=color_by, **kwargs)

    # Alias matching Monocle2 R names
    plot_cell_trajectory = plot_trajectory

    def plot_trajectory_overlay(self, ax, **kwargs):
        """Overlay the DDRTree principal-graph skeleton + branch points
        on an externally-drawn axes (e.g. one produced by
        ``ov.pl.embedding(basis='X_DDRTree')``).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes whose coordinate system matches
            ``adata.obsm['X_DDRTree']``.
        show_tree, show_branch_points : bool
            Toggle the MST edges and numbered branch-point markers.
        backbone_color, cell_link_size, branch_point_size,
        branch_point_color, branch_point_label_color,
        branch_point_label_fontsize, theta, x, y, zorder_base
            Forwarded to the underlying helper — see
            :func:`omicverse.external.monocle2_py.plotting.plot_trajectory_overlay`.

        Returns
        -------
        matplotlib.axes.Axes
            The same axes, for chaining.

        Examples
        --------
        Draw cells with omicverse's embedding plotter, then overlay the
        DDRTree backbone and branch points::

            import omicverse as ov
            fig, ax = plt.subplots(figsize=(7, 5))
            ov.pl.embedding(
                mono.adata, basis='X_DDRTree', color='State',
                ax=ax, show=False,
            )
            mono.plot_trajectory_overlay(ax)
        """
        return self._m2.plot_trajectory_overlay(self.adata, ax, **kwargs)

    def plot_trajectory_with_embedding(
        self,
        color: str = 'State',
        *,
        basis: str = 'X_DDRTree',
        cell_size: float = 4.0,
        figsize: tuple = (7, 5),
        show_tree: bool = True,
        show_branch_points: bool = True,
        backbone_color: str = 'black',
        cell_link_size: float = 0.75,
        branch_point_size: float = 150,
        **embedding_kwargs,
    ):
        """Convenience combo: render ``ov.pl.embedding`` then overlay
        the DDRTree backbone + branch points.

        This is the ``ov.pl.embedding(...)`` + ``plot_trajectory_overlay``
        pipeline in one call. For fine-grained control over the
        embedding (custom palette, multiple colour keys, subplots, …)
        call the two steps separately.

        Parameters
        ----------
        color : str
            Column to colour cells by — passed to ``ov.pl.embedding``.
        basis : str
            Key in ``adata.obsm`` that carries the trajectory layout.
            Default ``'X_DDRTree'``.
        cell_size : float
            Matplotlib scatter ``s`` equivalent for the cells. Mapped to
            ``ov.pl.embedding``'s ``size`` argument as ``cell_size ** 2``
            to match the R/Monocle convention where ``cell_size`` is a
            radius-like parameter.
        figsize, show_tree, show_branch_points, backbone_color,
        cell_link_size, branch_point_size
            Styling forwarded to the overlay.
        **embedding_kwargs
            Extra keyword arguments forwarded to ``ov.pl.embedding``.

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
        """
        import matplotlib.pyplot as plt
        import omicverse as ov

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ov.pl.embedding(
            self.adata,
            basis=basis,
            color=color,
            ax=ax,
            size=cell_size ** 2,
            show=False,
            **embedding_kwargs,
        )
        self._m2.plot_trajectory_overlay(
            self.adata, ax,
            show_tree=show_tree,
            show_branch_points=show_branch_points,
            backbone_color=backbone_color,
            cell_link_size=cell_link_size,
            branch_point_size=branch_point_size,
        )
        fig.tight_layout()
        return fig, ax

    def plot_complex_cell_trajectory(self, color_by: str = 'State', **kwargs):
        """Dendrogram-style trajectory layout (Pseudotime on Y-axis)."""
        return self._m2.plot_complex_cell_trajectory(self.adata, color_by=color_by, **kwargs)

    def plot_cell_clusters(self, color_by: str = 'Cluster', **kwargs):
        """Plot cells colored by cluster in reduced-dim space."""
        return self._m2.plot_cell_clusters(self.adata, color_by=color_by, **kwargs)

    def plot_genes_in_pseudotime(self, genes: List[str], **kwargs):
        """Gene expression vs pseudotime with smoothed curves."""
        return self._m2.plot_genes_in_pseudotime(self.adata, genes=genes, **kwargs)

    def plot_genes_branched_pseudotime(self, genes: List[str], branch_point: int = 1,
                                        **kwargs):
        """Gene expression split by branch."""
        return self._m2.plot_genes_branched_pseudotime(
            self.adata, genes=genes, branch_point=branch_point, **kwargs,
        )

    def plot_genes_branched_heatmap(self, branch_point: int = 1, **kwargs):
        """Heatmap of branch-specific gene expression."""
        return self._m2.plot_genes_branched_heatmap(
            self.adata, branch_point=branch_point, **kwargs,
        )

    def plot_multiple_branches_pseudotime(self, genes: List[str],
                                            branches: List, **kwargs):
        """Multi-branch gene expression curves."""
        return self._m2.plot_multiple_branches_pseudotime(
            self.adata, genes=genes, branches=branches, **kwargs,
        )

    def plot_multiple_branches_heatmap(self, branches: List, **kwargs):
        """Multi-branch expression heatmap."""
        return self._m2.plot_multiple_branches_heatmap(
            self.adata, branches=branches, **kwargs,
        )

    def plot_pseudotime_heatmap(self, genes: Optional[List[str]] = None, **kwargs):
        """Heatmap of gene expression sorted by pseudotime."""
        return self._m2.plot_pseudotime_heatmap(self.adata, genes=genes, **kwargs)

    def plot_genes_jitter(self, genes: List[str], grouping: str = 'State', **kwargs):
        """Jitter plot of gene expression by group."""
        return self._m2.plot_genes_jitter(self.adata, genes=genes, grouping=grouping, **kwargs)

    def plot_genes_violin(self, genes: List[str], grouping: str = 'State', **kwargs):
        """Violin plot of gene expression by group."""
        return self._m2.plot_genes_violin(self.adata, genes=genes, grouping=grouping, **kwargs)

    def plot_ordering_genes(self, **kwargs):
        """Dispersion vs mean-expression plot, highlighting ordering genes."""
        return self._m2.plot_ordering_genes(self.adata, **kwargs)

    def plot_pc_variance_explained(self, max_components: int = 50, **kwargs):
        """Plot variance explained by principal components."""
        return self._m2.plot_pc_variance_explained(
            self.adata, max_components=max_components, **kwargs,
        )

    def plot_rho_delta(self, **kwargs):
        """Plot rho vs delta for density-peak clustering."""
        return self._m2.plot_rho_delta(self.adata, **kwargs)

    # ------------------------------------------------------------------ #
    # Utility accessors
    # ------------------------------------------------------------------ #

    @property
    def pseudotime(self) -> Optional[pd.Series]:
        """Per-cell pseudotime (after order_cells)."""
        return self.adata.obs.get('Pseudotime')

    @property
    def state(self) -> Optional[pd.Series]:
        """Per-cell state (after order_cells)."""
        return self.adata.obs.get('State')

    @property
    def branch_points(self) -> list:
        """Branch-point vertex names in the learned tree."""
        return self.adata.uns.get('monocle', {}).get('branch_points', [])

    @property
    def Z(self) -> Optional[np.ndarray]:
        """Reduced-dim cell coordinates (dim × N)."""
        return self.adata.uns.get('monocle', {}).get('reducedDimS')

    @property
    def Y(self) -> Optional[np.ndarray]:
        """Tree-center coordinates (dim × K)."""
        return self.adata.uns.get('monocle', {}).get('reducedDimK')
