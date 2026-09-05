import numpy as np
import scipy.sparse as _sp
import scanpy as sc
import warnings
from ..pp import preprocess
from .._settings import add_reference
from .._registry import register_function


def _select_significant_svg_names(qvals, *, n_svgs, qval_threshold):
    """Return at most ``n_svgs`` finite q-values below the requested cutoff."""
    import pandas as pd

    values = pd.Series(qvals, copy=False).replace([np.inf, -np.inf], np.nan).dropna()
    if qval_threshold is not None:
        qval_threshold = float(qval_threshold)
        if not 0 <= qval_threshold <= 1:
            raise ValueError("qval_threshold must lie in [0, 1] or be None.")
        values = values[values < qval_threshold]
    if n_svgs is not None:
        n_svgs = int(n_svgs)
        if n_svgs < 0:
            raise ValueError("n_svgs must be non-negative or None.")
        values = values.nsmallest(n_svgs)
    else:
        values = values.sort_values()
    return values.index


def _adjust_testable_pvalues(values, method='fdr_bh'):
    from statsmodels.stats.multitest import multipletests
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & (values >= 0) & (values <= 1)
    adjusted = np.full(values.shape, np.nan)
    if valid.any():
        adjusted[valid] = multipletests(values[valid], method=method)[1]
    return adjusted


def _autocorr_library_subset(adata, indices):
    result = adata[indices].copy()
    result.uns.pop('spatial', None)
    return result


def _svg_multiple_libraries(adata, mode, n_svgs, target_sum, platform,
                            mt_startwith, library_key, selection, kwargs):
    import pandas as pd
    if mode not in ('moran', 'morani', 'somde', 'spatialde'):
        raise ValueError('Multi-library SVG currently supports Moran, SOMDE and SpatialDE; run other methods per library.')
    labels = adata.obs[library_key].astype(str).to_numpy()
    libraries = list(dict.fromkeys(labels))
    if len(libraries) != adata.obs[library_key].nunique():
        raise ValueError('Library labels collide after conversion to strings.')
    frames = []
    for library in libraries:
        subset = adata[labels == library].copy()
        subset.uns.pop('spatial', None)  # Independent coordinates; no image selection is used here.
        if subset.n_obs < 4:
            frame = pd.DataFrame({'gene': subset.var_names, 'statistic': np.nan, 'pvalue': np.nan})
        else:
            svg(subset, mode=mode, n_svgs=n_svgs, target_sum=target_sum, platform=platform,
                mt_startwith=mt_startwith, selection='top_n', **kwargs)
            stat, pval = (('moranI', 'moranI_pval') if mode in ('moran', 'morani')
                          else (f'{mode}_LLR', f'{mode}_pval'))
            if stat not in subset.var or pval not in subset.var:
                raise ValueError(f'{mode} backend did not provide raw statistics/p-values for global adjustment.')
            frame = pd.DataFrame({'gene': subset.var_names,
                                   'statistic': subset.var[stat].to_numpy(),
                                   'pvalue': subset.var[pval].to_numpy()})
        frame.insert(0, 'library', library)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table['qvalue'] = _adjust_testable_pvalues(table['pvalue'])
    table['testable'] = np.isfinite(table['pvalue']) & np.isfinite(table['statistic'])
    table['selected'] = False
    masks = pd.DataFrame(False, index=adata.var_names, columns=libraries)
    for library in libraries:
        eligible = table[(table.library == library) & table.testable]
        if selection == 'significant':
            eligible = eligible[eligible.qvalue < kwargs.get('qval_threshold', 0.05)]
            if mode in ('moran', 'morani'):
                eligible = eligible[eligible.statistic > 0]
        eligible = eligible.sort_values('statistic', ascending=False, kind='stable')
        if n_svgs is not None:
            eligible = eligible.head(n_svgs)
        table.loc[eligible.index, 'selected'] = True
        masks.loc[eligible.gene, library] = True
    table.index = table.index.astype(str)
    adata.uns['spatial_features_by_library'] = table
    adata.varm['space_variable_features_by_library'] = masks
    adata.var['space_variable_features'] = masks.any(axis=1)
    adata.var['highly_variable'] = adata.var['space_variable_features']
    adata.uns['space_svg_selection'] = {'method': mode, 'selection': selection,
        'correction_family': 'all_testable_library_gene_pairs', 'correction_method': 'fdr_bh',
        'global_mask': 'union_of_per_library_selected_candidates', 'library_key': library_key,
        'n_svgs_scope': 'per_library'}
    return adata


# ---------------------------------------------------------------------------
# Internal helpers for spatial autocorrelation
# ---------------------------------------------------------------------------

def _moran_i_scores(g, vals):
    """Compute Moran's I for each gene column in *vals*.

    Parameters
    ----------
    g : sparse (n, n) weight matrix (possibly row-normalised)
    vals : ndarray (n, n_genes)

    Returns
    -------
    scores : ndarray (n_genes,)
    """
    n = vals.shape[0]
    s0 = float(g.sum())
    x_dev = vals - vals.mean(axis=0, keepdims=True)   # (n, n_genes)
    g_x = g @ x_dev                                    # (n, n_genes)
    numerator   = (x_dev * g_x).sum(axis=0)
    denominator = (x_dev ** 2).sum(axis=0)
    return (n / s0) * numerator / np.maximum(denominator, 1e-15)


def _geary_c_scores(g, vals):
    """Compute Geary's C for each gene column in *vals*."""
    n = vals.shape[0]
    s0 = float(g.sum())
    row_sums = np.asarray(g.sum(axis=1)).ravel()
    col_sums = np.asarray(g.sum(axis=0)).ravel()
    x2       = vals ** 2
    term1    = (row_sums + col_sums) @ x2
    term2    = 2.0 * (vals * (g @ vals)).sum(axis=0)
    numerator   = term1 - term2
    mean        = vals.mean(axis=0, keepdims=True)
    denominator = ((vals - mean) ** 2).sum(axis=0)
    return ((n - 1) / (2.0 * s0)) * numerator / np.maximum(denominator, 1e-15)


def _analytic_pval(scores, g, mode, n, two_tailed):
    """Analytical p-values under the normal approximation."""
    from scipy import stats

    s0  = float(g.sum())
    g_sym = g + g.T
    if _sp.issparse(g_sym):
        s1 = 0.5 * float(g_sym.multiply(g_sym).sum())
    else:
        s1 = 0.5 * float((g_sym ** 2).sum())
    row_sums = np.asarray(g.sum(axis=1)).ravel()
    col_sums = np.asarray(g.sum(axis=0)).ravel()
    s2  = float(np.sum((row_sums + col_sums) ** 2))
    s02 = s0 ** 2

    if mode == 'moran':
        expected = -1.0 / (n - 1)
        v_num  = n ** 2 * s1 - n * s2 + 3 * s02
        v_den  = (n - 1) * (n + 1) * s02
        var_sc = v_num / v_den - expected ** 2
        z = (scores - expected) / np.sqrt(np.maximum(var_sc, 1e-15))
        pvals = stats.norm.sf(np.abs(z)) * 2 if two_tailed else stats.norm.sf(z)
    else:  # geary
        expected = 1.0
        v_num  = (2 * s1 + s2) * (n - 1) - 4 * s02
        v_den  = 2 * (n + 1) * s02
        var_sc = v_num / v_den
        z = (scores - expected) / np.sqrt(np.maximum(var_sc, 1e-15))
        pvals = stats.norm.sf(np.abs(z)) * 2 if two_tailed else stats.norm.cdf(z)

    return pvals


# ---------------------------------------------------------------------------
# Public spatial graph / autocorrelation functions
# ---------------------------------------------------------------------------


def _spatial_distance_graph(
    coords,
    *,
    n_neighs,
    radius,
    delaunay,
    set_diag,
    coord_type,
):
    """Build one within-library symmetric spatial distance graph."""
    from scipy.spatial import Delaunay, QhullError
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=np.float64)
    n_obs = coords.shape[0]
    coord_type = str(coord_type).lower()
    if coord_type not in {'generic', 'grid'}:
        raise ValueError("coord_type must be 'generic' or 'grid'.")

    if isinstance(radius, (tuple, list)):
        if len(radius) != 2:
            raise ValueError("radius must be a number or a (min_radius, max_radius) pair.")
        r_min, r_max = map(float, radius)
    elif radius is None:
        r_min, r_max = 0.0, None
    else:
        r_min, r_max = 0.0, float(radius)
    if r_min < 0 or (r_max is not None and r_max <= 0) or (
        r_max is not None and r_min > r_max
    ):
        raise ValueError("radius bounds must satisfy 0 <= min_radius <= max_radius and max_radius > 0.")

    if delaunay:
        if n_obs < coords.shape[1] + 1:
            raise ValueError(
                "Delaunay triangulation requires at least n_dims + 1 observations "
                "within every library. "
                f"Got {n_obs} observations with {coords.shape[1]} spatial dimensions."
            )
        try:
            tri = Delaunay(coords)
        except QhullError as exc:
            raise ValueError(
                "Failed to build a Delaunay graph from spatial coordinates. "
                "Check for duplicated/degenerate coordinates within each library."
            ) from exc

        edge_pairs = []
        for simplex in np.asarray(tri.simplices, dtype=np.int64):
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    a, b = sorted((int(simplex[i]), int(simplex[j])))
                    if a != b:
                        edge_pairs.append((a, b))
        if edge_pairs:
            edges = np.unique(np.asarray(edge_pairs, dtype=np.int64), axis=0)
            distances = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
            row = np.concatenate([edges[:, 0], edges[:, 1]])
            col = np.concatenate([edges[:, 1], edges[:, 0]])
            data = np.concatenate([distances, distances])
            dist_mat = _sp.coo_matrix((data, (row, col)), shape=(n_obs, n_obs)).tocsr()
        else:
            dist_mat = _sp.csr_matrix((n_obs, n_obs), dtype=np.float64)
        if r_max is not None:
            outside = (dist_mat.data < r_min) | (dist_mat.data > r_max)
            dist_mat.data[outside] = 0.0
            dist_mat.eliminate_zeros()
    elif r_max is not None:
        nn = NearestNeighbors(algorithm='ball_tree', radius=r_max)
        nn.fit(coords)
        dist_mat = nn.radius_neighbors_graph(coords, radius=r_max, mode='distance')
        if r_min > 0:
            dist_mat.data[dist_mat.data < r_min] = 0.0
            dist_mat.eliminate_zeros()
    else:
        n_neighs = int(n_neighs)
        if n_neighs <= 0:
            raise ValueError("n_neighs must be a positive integer.")
        # sklearn counts the query observation itself when the fitted coordinates
        # are passed explicitly. Ask for one extra entry, then drop the diagonal.
        query_n = min(n_neighs + 1, n_obs)
        nn = NearestNeighbors(n_neighbors=query_n, algorithm='ball_tree')
        nn.fit(coords)
        dist_mat = nn.kneighbors_graph(coords, n_neighbors=query_n, mode='distance')
        dist_mat.setdiag(0)
        dist_mat.eliminate_zeros()

        if coord_type == 'grid' and dist_mat.nnz:
            step = float(np.median(dist_mat.data))
            dist_mat.data[dist_mat.data > step * 1.4] = 0.0
            dist_mat.eliminate_zeros()

    if not set_diag:
        dist_mat.setdiag(0)
        dist_mat.eliminate_zeros()

    return dist_mat.maximum(dist_mat.T).tocsr()


@register_function(
    aliases=["空间邻域图", "spatial_neighbors", "空间邻居", "构建空间图"],
    category="space",
    description="Build a spatial neighborhood graph (KNN or radius-based) from obsm['spatial'] coordinates",
    examples=[
        "# Default: 6 nearest neighbours",
        "ov.space.spatial_neighbors(adata, n_neighs=6)",
        "# Radius-based neighbours",
        "ov.space.spatial_neighbors(adata, radius=200)",
        "# Custom number of neighbours",
        "ov.space.spatial_neighbors(adata, n_neighs=8, key_added='spatial')",
        "# Concatenated slides: never connect across libraries",
        "ov.space.spatial_neighbors(adata, n_neighs=6, library_key='slice')",
    ],
    related=["space.spatial_autocorr", "space.moranI", "space.svg"],
)
def spatial_neighbors(
    adata,
    spatial_key: str = 'spatial',
    n_neighs: int = 6,
    radius=None,
    delaunay: bool = False,
    set_diag: bool = False,
    key_added: str = 'spatial',
    coord_type: str = 'generic',
    copy: bool = False,
    library_key=None,
):
    r"""Build a spatial neighborhood graph from coordinates stored in ``adata.obsm``.

    The resulting connectivity and distance matrices are stored in
    ``adata.obsp['{key_added}_connectivities']`` and
    ``adata.obsp['{key_added}_distances']``.  Graph metadata is written to
    ``adata.uns['{key_added}_neighbors']``.

    Arguments:
        adata: AnnData object with spatial coordinates in ``adata.obsm[spatial_key]``.
        spatial_key: Key in ``adata.obsm`` that stores 2-D spatial coordinates. Default: 'spatial'.
        n_neighs: Number of nearest spatial neighbors (used when *radius* is ``None`` and
            ``delaunay=False``). Default: 6.
        radius: Radius (or ``(min_radius, max_radius)`` tuple) for radius-based graph.
            When set, *n_neighs* is ignored. Default: None.
        delaunay: Whether to build the graph from a Delaunay triangulation of the
            spatial coordinates. When set, *n_neighs* is ignored. Default: False.
        set_diag: Whether to include self-loops in the connectivity matrix. Default: False.
        coord_type: ``'generic'`` (default) keeps every k-nearest neighbour. ``'grid'``
            additionally drops edges longer than 1.4 lattice steps, which is what you
            want on an array platform such as Visium or Stereo-seq: without it, spots
            on the rim of the tissue reach across the gap to fill their k quota and
            acquire neighbours they do not touch. On a 400-spot Visium subset the
            generic graph gives node degrees of 5-10 where the hexagonal lattice
            allows at most 6. The default is ``'generic'`` so that existing results do
            not change silently; pass ``'grid'`` for array data, and note that
            :func:`omicverse.space.sepal` assumes a lattice and needs it.
        key_added: Prefix for the keys added to ``adata.obsp`` and ``adata.uns``. Default: 'spatial'.
        copy: If ``True``, return ``(connectivities, distances)`` as sparse matrices
            without modifying ``adata``. Default: False.
        library_key: Optional column in ``adata.obs`` identifying independent
            slides/libraries. When provided, neighbors are built within each
            library so overlapping coordinate systems cannot create cross-slide
            edges. Default: None.

    Returns:
        None or (connectivities, distances): Modifies *adata* in-place when
        *copy* is ``False``. Returns matrices without modifying *adata* when
        *copy* is ``True``.

    Examples:
        >>> import omicverse as ov
        >>> ov.space.spatial_neighbors(adata, n_neighs=6)
        >>> # radius graph
        >>> ov.space.spatial_neighbors(adata, radius=150)
    """
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n_obs  = coords.shape[0]
    coord_type = str(coord_type).lower()
    if n_obs == 0:
        raise ValueError("spatial_neighbors requires at least one observation.")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must be a 2-D coordinate matrix with "
            f"at least two columns; got shape {coords.shape}."
        )
    if not np.isfinite(coords).all():
        raise ValueError(f"adata.obsm[{spatial_key!r}] contains NaN or infinite coordinates.")

    library_sizes = None
    if library_key is None:
        spatial_meta = adata.uns.get('spatial', {})
        if isinstance(spatial_meta, dict) and len(spatial_meta) > 1:
            warnings.warn(
                "Multiple spatial libraries are present but `library_key` was not "
                "provided. If their coordinate systems overlap, the graph may contain "
                "cross-library edges.",
                UserWarning,
                stacklevel=2,
            )
        dist_mat = _spatial_distance_graph(
            coords,
            n_neighs=n_neighs,
            radius=radius,
            delaunay=delaunay,
            set_diag=set_diag,
            coord_type=coord_type,
        )
    else:
        if library_key not in adata.obs:
            raise KeyError(f"library_key {library_key!r} was not found in adata.obs.")
        library_values = adata.obs[library_key]
        if library_values.isna().any():
            raise ValueError(f"adata.obs[{library_key!r}] contains missing library labels.")

        rows = []
        cols = []
        values = []
        library_sizes = {}
        labels = library_values.astype(str).to_numpy()
        if len(set(labels)) != library_values.nunique():
            raise ValueError('Library labels collide after conversion to strings.')
        for library in dict.fromkeys(labels):
            obs_idx = np.flatnonzero(labels == library)
            library_sizes[str(library)] = int(len(obs_idx))
            sub_dist = _spatial_distance_graph(
                coords[obs_idx],
                n_neighs=n_neighs,
                radius=radius,
                delaunay=delaunay,
                set_diag=set_diag,
                coord_type=coord_type,
            ).tocoo()
            rows.append(obs_idx[sub_dist.row])
            cols.append(obs_idx[sub_dist.col])
            values.append(sub_dist.data)

        row = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
        col = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
        data = np.concatenate(values) if values else np.array([], dtype=np.float64)
        dist_mat = _sp.coo_matrix((data, (row, col)), shape=(n_obs, n_obs)).tocsr()

    conn_mat = (dist_mat > 0).astype(np.float64).tocsr()

    if set_diag:
        conn_mat.setdiag(1.0)

    avg_deg = conn_mat.nnz / n_obs
    print(f"Spatial neighbors: {n_obs} cells, {conn_mat.nnz} connections "
          f"(avg {avg_deg:.1f} neighbors/cell).")

    if copy:
        return conn_mat, dist_mat

    adata.obsp[f'{key_added}_connectivities'] = conn_mat
    adata.obsp[f'{key_added}_distances']      = dist_mat
    adata.uns[f'{key_added}_neighbors'] = {
        'connectivities_key': f'{key_added}_connectivities',
        'distances_key':      f'{key_added}_distances',
        'params': {
            'n_neighbors': n_neighs,
            'radius':      radius,
            'delaunay':    delaunay,
            'method':      'spatial',
            'spatial_key': spatial_key,
            'coord_type':  coord_type,
            'library_key': library_key,
            'library_sizes': library_sizes,
        },
    }
    print(f"Stored in adata.obsp['{key_added}_connectivities'] "
          f"and adata.obsp['{key_added}_distances'].")
    add_reference(adata, 'omicverse', 'spatial neighborhood graph construction')


@register_function(
    aliases=["空间自相关", "spatial_autocorr", "莫兰指数计算", "moran_geary"],
    category="space",
    description="Compute Moran's I or Geary's C spatial autocorrelation for gene expression",
    examples=[
        "# Moran's I for all genes (after spatial_neighbors)",
        "ov.space.spatial_neighbors(adata, n_neighs=6)",
        "df = ov.space.spatial_autocorr(adata, mode='moran')",
        "# Geary's C with permutation p-values",
        "df = ov.space.spatial_autocorr(adata, mode='geary', n_perms=1000)",
        "# Specific genes only",
        "df = ov.space.spatial_autocorr(adata, genes=svgs, mode='moran')",
    ],
    related=["space.spatial_neighbors", "space.moranI", "space.svg"],
)
def spatial_autocorr(
    adata,
    connectivity_key: str = 'spatial_connectivities',
    genes=None,
    mode: str = 'moran',
    transformation: bool = True,
    n_perms=None,
    two_tailed: bool = False,
    corr_method='fdr_bh',
    layer=None,
    seed=None,
    copy: bool = False,
    n_jobs: int = 1,
    library_key=None,
    _connectivity=None,
):
    r"""Compute spatial autocorrelation statistics for gene expression.

    Moran's I (``mode='moran'``) measures positive spatial autocorrelation on
    ``(-1, 1]``; Geary's C (``mode='geary'``) measures the opposite – values
    near 0 indicate strong clustering.  P-values are computed analytically
    under the normal approximation and optionally via label permutation.

    Arguments:
        adata: AnnData with a spatial connectivity matrix in ``adata.obsp``.
        connectivity_key: Key of the spatial connectivity matrix in ``adata.obsp``. Default: 'spatial_connectivities'.
        genes: Gene names or indices to test.  ``None`` tests all genes. Default: None.
        mode: ``'moran'`` (Moran's I) or ``'geary'`` (Geary's C). Default: 'moran'.
        transformation: Row-normalise the connectivity matrix before scoring. Default: True.
        n_perms: Number of label-permutation iterations for empirical p-values.
            ``None`` uses only the analytical p-value. Default: None.
        two_tailed: Use two-tailed test for the normal-approximation z-score. Default: False.
        corr_method: Multiple-testing correction method passed to
            ``statsmodels.stats.multitest.multipletests`` (e.g. ``'fdr_bh'``). Default: 'fdr_bh'.
        layer: Expression layer to use.  ``None`` uses ``adata.X``. Default: None.
        seed: Random seed for permutation testing. Default: None.
        copy: Return the result DataFrame instead of (also) storing it in ``adata.uns``. Default: False.
        n_jobs: Reserved for future parallel permutation support. Default: 1.
        library_key: Column identifying independent libraries. Multiple libraries
            are tested separately and returned as a long table with library/gene
            columns, with correction across all testable library-gene pairs.
            Constant genes and slices with fewer than four spots or no edges are
            untestable (NaN statistics/p-values), not evidence of significance.

    Returns:
        DataFrame: Results with columns ``I`` / ``C``, ``pval_norm``, and optionally
        ``pval_sim``, ``pval_z_sim``, ``pval_adj``.  Also stored in
        ``adata.uns['moranI']`` or ``adata.uns['gearyC']``.

    Examples:
        >>> import omicverse as ov
        >>> ov.space.spatial_neighbors(adata, n_neighs=6)
        >>> df = ov.space.spatial_autocorr(adata, mode='moran')
        >>> df.head()
    """
    import pandas as pd
    from sklearn.preprocessing import normalize

    if library_key is None:
        graph_key = connectivity_key.removesuffix('_connectivities') + '_neighbors'
        library_key = adata.uns.get(graph_key, {}).get('params', {}).get('library_key')
        if library_key is None and len(adata.uns.get('spatial', {})) > 1:
            raise ValueError('Multiple spatial libraries require an explicit library_key for inference.')

    if _connectivity is None and connectivity_key not in adata.obsp:
        raise KeyError(
            f"'{connectivity_key}' not found in adata.obsp. "
            "Run ov.space.spatial_neighbors(adata) first."
        )
    if mode not in ('moran', 'geary'):
        raise ValueError(f"mode must be 'moran' or 'geary', got '{mode}'")

    # ---- weight matrix -----------------------------------------------
    source_graph = (
        adata.obsp[connectivity_key]
        if _connectivity is None
        else _connectivity
    )
    if library_key is not None:
        if library_key not in adata.obs or adata.obs[library_key].isna().any():
            raise ValueError('library_key must identify every observation without missing labels.')
        labels = adata.obs[library_key].astype(str).to_numpy()
        graph = _sp.csr_matrix(source_graph)
        edges = graph.tocoo()
        if np.any(labels[edges.row] != labels[edges.col]):
            raise ValueError('Connectivity graph contains cross-library edges; rebuild with library_key.')
        libraries = list(dict.fromkeys(labels))
        if len(libraries) > 1:
            frames = []
            for library in libraries:
                indices = np.flatnonzero(labels == library)
                result = spatial_autocorr(
                    _autocorr_library_subset(adata, indices), genes=genes, mode=mode, transformation=transformation,
                    n_perms=n_perms, two_tailed=two_tailed, corr_method=None, layer=layer,
                    seed=seed, copy=True, n_jobs=n_jobs,
                    _connectivity=graph[indices][:, indices],
                )
                result.insert(0, 'gene', result.index.astype(str))
                result.insert(0, 'library', library)
                frames.append(result.reset_index(drop=True))
            result = pd.concat(frames, ignore_index=True)
            if corr_method is not None:
                result['pval_adj'] = _adjust_testable_pvalues(
                    result['pval_sim' if n_perms is not None else 'pval_norm'], corr_method)
            result.index = result.index.astype(str)
            if not copy:
                adata.uns['moranI' if mode == 'moran' else 'gearyC'] = result
            return result
        library_key = None
    if source_graph.shape != (adata.n_obs, adata.n_obs):
        raise ValueError(
            "The spatial connectivity matrix must have shape "
            f"({adata.n_obs}, {adata.n_obs}); got {source_graph.shape}."
        )
    g = source_graph.astype(np.float64).copy()
    if transformation:
        g = normalize(g, norm='l1', axis=1)

    # ---- gene selection ----------------------------------------------
    if genes is None:
        genes = adata.var_names.tolist()
    elif isinstance(genes, (str, int)):
        genes = [genes]
    genes = list(genes)

    gene_idx = adata.var_names.get_indexer(genes)
    if (gene_idx == -1).any():
        bad = [gn for gn, i in zip(genes, gene_idx) if i == -1]
        raise ValueError(f"Genes not found in adata.var_names: {bad[:5]}")

    # ---- expression matrix (n_obs × n_genes) -------------------------
    mat = adata.layers[layer] if layer is not None else adata.X
    vals = mat[:, gene_idx]
    if _sp.issparse(vals):
        vals = vals.toarray()
    vals = np.asarray(vals, dtype=np.float64)

    n = adata.n_obs
    if n < 4 or float(g.sum()) <= 0:
        stat_key, uns_key = ('I', 'moranI') if mode == 'moran' else ('C', 'gearyC')
        result = pd.DataFrame({stat_key: np.nan, 'pval_norm': np.nan,
                               'testable': False}, index=genes)
        if n_perms is not None:
            result['pval_sim'] = np.nan
        if corr_method is not None:
            result['pval_adj'] = np.nan
        if not copy:
            adata.uns[uns_key] = result
        return result
    library_codes = None

    # ---- scores ------------------------------------------------------
    if mode == 'moran':
        scores   = _moran_i_scores(g, vals)
        stat_key = 'I'
        uns_key  = 'moranI'
    else:
        scores   = _geary_c_scores(g, vals)
        stat_key = 'C'
        uns_key  = 'gearyC'

    if library_codes is None:
        pvals_norm = _analytic_pval(scores, g, mode, n, two_tailed)
    else:
        pvals_norm = np.full(len(genes), np.nan, dtype=np.float64)

    df = pd.DataFrame({stat_key: scores, 'pval_norm': pvals_norm}, index=genes)
    testable = np.isfinite(vals).all(axis=0) & (np.var(vals, axis=0) > 0)
    df['testable'] = testable

    # ---- permutation p-values ----------------------------------------
    if n_perms is not None:
        if not isinstance(n_perms, (int, np.integer)) or n_perms < 1:
            raise ValueError("n_perms must be a positive integer or None.")
        rng = np.random.default_rng(seed)
        perm_scores = np.empty((n_perms, len(genes)))
        for i in range(n_perms):
            if library_codes is None:
                perm_idx = rng.permutation(n)
            else:
                perm_idx = np.arange(n)
                for code in np.unique(library_codes):
                    group_idx = np.flatnonzero(library_codes == code)
                    perm_idx[group_idx] = rng.permutation(group_idx)
            v_perm   = vals[perm_idx]
            perm_scores[i] = (
                _moran_i_scores(g, v_perm) if mode == 'moran'
                else _geary_c_scores(g, v_perm)
            )
        if two_tailed:
            expected = -1.0 / (n - 1) if mode == 'moran' else 1.0
            exceedances = np.sum(
                np.abs(perm_scores - expected) >= np.abs(scores - expected)[np.newaxis, :], axis=0
            )
        elif mode == 'moran':
            exceedances = np.sum(perm_scores >= scores[np.newaxis, :], axis=0)
        else:
            exceedances = np.sum(perm_scores <= scores[np.newaxis, :], axis=0)
        df['pval_sim'] = (exceedances + 1.0) / (n_perms + 1.0)
        perm_std = perm_scores.std(axis=0)
        df['z_sim'] = (
            (scores - perm_scores.mean(axis=0)) / np.maximum(perm_std, 1e-10)
        )
        from scipy.stats import norm
        df['pval_z_sim'] = (2 * norm.sf(np.abs(df['z_sim'])) if two_tailed
                            else norm.sf(df['z_sim']) if mode == 'moran' else norm.cdf(df['z_sim']))

    df.loc[~testable, [col for col in df if col != 'testable']] = np.nan
    # ---- multiple-testing correction ---------------------------------
    if corr_method is not None:
        from statsmodels.stats.multitest import multipletests
        pval_col = 'pval_sim' if (n_perms is not None) else 'pval_norm'
        pvals_adj = _adjust_testable_pvalues(df[pval_col], corr_method)
        df['pval_adj'] = pvals_adj

    df = df.sort_values(stat_key, ascending=(mode == 'geary'))

    if not copy:
        adata.uns[uns_key] = df
        print(f"Stored {len(genes)} gene results in adata.uns['{uns_key}'].")
        add_reference(adata, 'omicverse', f'spatial autocorrelation ({mode}) analysis')

    return df


@register_function(
    aliases=["莫兰指数", "moran_i", "moranI", "空间莫兰", "Moran's I"],
    category="space",
    description="Compute Moran's I spatial autocorrelation (convenience wrapper for spatial_autocorr)",
    examples=[
        "# Build graph then compute Moran's I",
        "ov.space.spatial_neighbors(adata, n_neighs=6)",
        "df = ov.space.moranI(adata)",
        "# With permutation p-values",
        "df = ov.space.moranI(adata, n_perms=1000, seed=42)",
        "# Auto-build spatial graph and run Moran's I in one call",
        "df = ov.space.moranI(adata, auto_spatial_neighbors=True, n_neighs=6)",
        "# Concatenated slides: build neighbors within each library",
        "df = ov.space.moranI(adata, auto_spatial_neighbors=True, library_key='slice')",
        "# Subset to pre-selected SVGs",
        "svgs = adata.var_names[adata.var['space_variable_features']]",
        "df = ov.space.moranI(adata, genes=svgs)",
    ],
    related=["space.spatial_neighbors", "space.spatial_autocorr", "space.svg"],
)
def moranI(
    adata,
    connectivity_key: str = 'spatial_connectivities',
    genes=None,
    transformation: bool = True,
    n_perms=None,
    two_tailed: bool = False,
    corr_method='fdr_bh',
    layer=None,
    seed=None,
    copy: bool = False,
    auto_spatial_neighbors: bool = False,
    n_neighs: int = 6,
    radius=None,
    spatial_key: str = 'spatial',
    library_key=None,
):
    r"""Compute Moran's I spatial autocorrelation for gene expression.

    A convenience wrapper around :func:`spatial_autocorr` with ``mode='moran'``.
    Set *auto_spatial_neighbors* to ``True`` to build the spatial neighborhood
    graph automatically via :func:`spatial_neighbors` when the connectivity matrix
    is missing from ``adata.obsp``.

    Arguments:
        adata: AnnData with spatial coordinates in ``adata.obsm`` and optionally a
            precomputed connectivity in ``adata.obsp``.
        connectivity_key: Key of the spatial connectivity matrix in ``adata.obsp``. Default: 'spatial_connectivities'.
        genes: Gene names/indices to test.  ``None`` tests all genes. Default: None.
        transformation: Row-normalise the weight matrix before scoring. Default: True.
        n_perms: Permutations for empirical p-values; ``None`` uses only the analytical value. Default: None.
        two_tailed: Two-tailed z-score test. Default: False.
        corr_method: Multiple-testing correction (``'fdr_bh'``, ``'bonferroni'``, …). Default: 'fdr_bh'.
        layer: Expression layer to use; ``None`` uses ``adata.X``. Default: None.
        seed: Random seed for permutation reproducibility. Default: None.
        copy: Return the result DataFrame without modifying ``adata``. This also
            keeps an automatically constructed graph off the input object.
            Default: False.
        auto_spatial_neighbors: Automatically build the spatial neighborhood graph if
            *connectivity_key* is absent from ``adata.obsp``. Default: False.
        n_neighs: Number of KNN neighbours used when *auto_spatial_neighbors* is ``True``. Default: 6.
        radius: Radius for radius-based graph when *auto_spatial_neighbors* is ``True``. Default: None.
        spatial_key: Key in ``adata.obsm`` holding 2-D coordinates. Default: 'spatial'.
        library_key: Optional column in ``adata.obs`` identifying independent
            slides. Forwarded to :func:`spatial_neighbors` when the graph is
            built automatically, preventing cross-library edges. Default: None.

    Returns:
        DataFrame: Moran's I results with columns ``I``, ``pval_norm``, and optionally
        ``pval_sim``, ``pval_z_sim``, ``pval_adj``.  Also stored in ``adata.uns['moranI']``.

    Examples:
        >>> import omicverse as ov
        >>> ov.space.spatial_neighbors(adata, n_neighs=6)
        >>> df = ov.space.moranI(adata)
        >>> df.head()
        >>> # One-liner with auto graph building
        >>> df = ov.space.moranI(adata, auto_spatial_neighbors=True)
    """
    connectivity = None
    if auto_spatial_neighbors and connectivity_key not in adata.obsp:
        print(f"'{connectivity_key}' not found – building spatial neighbors …")
        graph_result = spatial_neighbors(
            adata,
            spatial_key=spatial_key,
            n_neighs=n_neighs,
            radius=radius,
            key_added=connectivity_key.replace('_connectivities', ''),
            library_key=library_key,
            copy=copy,
        )
        if copy:
            connectivity, _ = graph_result

    return spatial_autocorr(
        adata,
        connectivity_key=connectivity_key,
        genes=genes,
        mode='moran',
        transformation=transformation,
        n_perms=n_perms,
        two_tailed=two_tailed,
        corr_method=corr_method,
        layer=layer,
        seed=seed,
        copy=copy,
        library_key=library_key,
        _connectivity=connectivity,
    )

@register_function(
    aliases=["空间变异基因", "svg", "spatially_variable_genes", "空间变异基因检测", "SVG检测"],
    category="space",
    description="Identify spatially variable genes using PROST, Moran's I, Spateo, SOMDE, or SpatialDE; Pearson-residual HVGs remain as a non-spatial compatibility prefilter",
    prerequisites={
        'optional_functions': []
    },
    requires={
        'obsm': ['spatial']  # Spatial coordinates required
    },
    produces={
        'var': ['space_variable_features']
    },
    auto_fix='none',
    examples=[
        "# Basic SVG detection with PROST",
        "adata = ov.space.svg(adata, mode='prost', n_svgs=3000)",
        "# Non-spatial Pearson-residual HVG prefilter (compatibility mode)",
        "adata = ov.space.svg(adata, mode='pearson_residuals', n_svgs=2000)",
        "# High-resolution analysis",
        "adata = ov.space.svg(adata, mode='prost', n_svgs=5000,",
        "                     target_sum=1e5, platform='visium')",
        "# Using SOMDE (SOM-accelerated SpatialDE)",
        "adata = ov.space.svg(adata, mode='somde', k=20)",
        "# SOMDE with custom threshold and extra training",
        "adata = ov.space.svg(adata, mode='somde', k=20, qval_threshold=0.05, retrain_epoch=100)",
        "# Using SpatialDE (GP-based, direct on single cells)",
        "adata = ov.space.svg(adata, mode='spatialde', qval_threshold=0.05)",
        "# SpatialDE with custom gene filter and regress formula",
        "adata = ov.space.svg(adata, mode='spatialde', min_total_count=3,",
        "                     regress_formula='np.log(total_counts)')",
        "# Access identified SVGs",
        "svgs = adata.var_names[adata.var['space_variable_features']]"
    ],
    related=["pp.preprocess", "space.clusters", "space.pySTAGATE"]
)
def svg(adata,mode='prost',n_svgs=3000,target_sum=50*1e4,platform="visium",
        mt_startwith='MT-',library_key=None,selection='significant',**kwargs):
    # somde kwargs: k=20, qval_threshold=0.05, retrain_epoch=0
    r"""Identify spatially variable genes using multiple methods.
    
    This function identifies genes that show significant spatial variation in their
    expression patterns across the tissue. It supports spatial methods including
    PROST, Moran's I, Spateo, SOMDE, and SpatialDE. A clearly marked
    Pearson-residual non-spatial prefilter is retained for compatibility.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData containing expression matrix and coordinates in
        ``adata.obsm['spatial']``.
    mode : {'prost', 'moran', 'spateo', 'somde', 'spatialde', 'pearson_residuals'}, default='prost'
        SVG detection backend. ``'pearson_residuals'`` is retained as a
        non-spatial HVG prefilter for compatibility; it does not use coordinates
        and must not be interpreted as an SVG significance test. The legacy
        spelling ``'pearsonr'`` is an alias for that prefilter.
    selection : {'significant', 'top_n'}, default='significant'
        For Moran, SOMDE and SpatialDE, select significant genes before applying
        the count cap. Use top_n explicitly for the legacy ranked prefilter.
        Multiple libraries are tested separately with BH across all testable
        library-gene pairs; the global mask is a union of selected candidates.
    n_svgs : int, default=3000
        Maximum number of genes to select. Significance-based methods may return
        fewer when fewer genes pass ``qval_threshold``.
    target_sum : float, default=50*1e4
        Target-sum used during normalization.
    platform : str, default='visium'
        Platform identifier used by PROST preprocessing.
    mt_startwith : str, default='MT-'
        Mitochondrial gene prefix excluded by default.
    library_key : str or None, default=None
        Observation column identifying independent slides when ``mode='moran'``.
        The automatically constructed spatial graph is then block-diagonal by
        library, so overlapping local coordinate systems cannot create
        cross-slide edges.
    **kwargs
        Additional method-specific options.
        For ``somde``: ``k``, ``qval_threshold``, ``retrain_epoch``.
        For ``spatialde``: ``qval_threshold``, ``min_total_count``, ``regress_formula``,
        ``n_jobs``, ``kernel_space``, ``approx_rank``, ``approx_seed``, ``approx_models``,
        ``show_progress``.

    Returns
    -------
    AnnData
        Updated AnnData with SVG flags in
        ``adata.var['space_variable_features']`` and ``adata.var['highly_variable']``.

    Notes:
        - PROST mode requires opencv-python package
        - Different modes use different statistical approaches:
            - PROST: Pattern recognition and spatial autocorrelation
            - pearson_residuals: non-spatial Pearson-residual HVG prefilter
            - spateo: Wasserstein distance-based spatial variation
            - somde: SOM-compressed SpatialDE GP test (fast for large datasets)
        - SOMDE kwargs: ``k`` (cells/node, default 20), ``qval_threshold`` (default 0.05),
          ``retrain_epoch`` (extra SOM epochs, default 0)
        - SOMDE stores ``adata.var['somde_LLR']``, ``somde_pval``, ``somde_qval``, ``somde_FSV``
        - SOMDE requires ``somoclu`` and ``patsy``
        - spatialde: GP-based test directly on single-cell coordinates (no SOM compression),
          uses bundled NaiveDE + SpatialDE packages from ``omicverse/external/``
        - SpatialDE kwargs: ``qval_threshold`` (default 0.05), ``min_total_count`` (default 3),
          ``regress_formula`` (default ``'np.log(total_counts)'``), ``n_jobs`` (default ``1``),
          ``kernel_space`` (optional custom covariance search space), ``approx_rank`` (optional
          Nyström low-rank approximation rank), ``approx_seed`` (landmark sampling seed),
          ``approx_models`` (kernel list for approximation, default ``('SE',)``),
          ``show_progress`` (whether to show tqdm progress bars, default ``True``)
        - SpatialDE stores ``adata.var['spatialde_LLR']``, ``spatialde_pval``, ``spatialde_qval``,
          ``spatialde_FSV``, ``spatialde_l``
        - SpatialDE requires ``patsy`` and ``tqdm``
        - Mitochondrial genes are excluded by default
        - Results are normalized and log-transformed

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load spatial data
        >>> adata = sc.read_visium(...)
        >>> # Find SVGs using PROST
        >>> adata = ov.space.svg(
        ...     adata,
        ...     mode='prost',
        ...     n_svgs=2000,
        ...     platform='visium'
        ... )
        >>> # Access SVGs
        >>> svgs = adata.var_names[adata.var['space_variable_features']]
    """
    import numpy as np
    mode = str(mode).lower()
    if n_svgs is not None and (isinstance(n_svgs, bool) or not isinstance(n_svgs, (int, np.integer)) or n_svgs < 0):
        raise ValueError('n_svgs must be a non-negative integer or None.')
    threshold = kwargs.get('qval_threshold', 0.05)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('qval_threshold must lie in [0, 1].')
    if selection not in ('significant', 'top_n'):
        raise ValueError("selection must be 'significant' or 'top_n'.")
    if library_key is not None:
        if library_key not in adata.obs or adata.obs[library_key].isna().any():
            raise ValueError('library_key must exist without missing labels.')
        if adata.obs[library_key].nunique() > 1:
            return _svg_multiple_libraries(adata, mode, n_svgs, target_sum, platform,
                                           mt_startwith, library_key, selection, kwargs)

    if mode=='prost':
        from ..external.PROST import prepare_for_PI,cal_PI,spatial_autocorrelation,feature_selection

        if 'counts' not in adata.layers.keys():
            adata.layers['counts'] = adata.X.copy()
        # Calculate PI
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PROST SVG detection requires OpenCV. Install it with "
                "`pip install opencv-python`."
            ) from exc
        bdata=adata.copy()
        bdata = prepare_for_PI(bdata, platform=platform)
        bdata = cal_PI(bdata, platform=platform)
        print('PI calculation is done!')

        # Spatial autocorrelation test
        #spatial_autocorrelation(adata)
        #print('Spatial autocorrelation test is done!')

        # Remove MT-gene
        
        #drop_gene_name = mt_startwith
        #selected_gene_name=list(adata.var_names[adata.var_names.str.contains(mt_startwith)==False])
        sc.pp.normalize_total(bdata, target_sum=target_sum)
        sc.pp.log1p(bdata)
        #print('normalization and log1p are done!')
        #adata.raw = adata
        bdata = feature_selection(bdata, 
                                  by = mode, n_top_genes = n_svgs)
        adata.var['space_variable_features'] = bdata.var['space_variable_features']
        adata.var['SEP'] = bdata.var['SEP']
        adata.var['SIG'] = bdata.var['SIG']
        adata.var['PI'] = bdata.var['PI']
        add_reference(adata,'PROST','spatial variable gene selection with PROST')
        #print(f'{n_svgs} SVGs are selected!')
    elif mode in {'pearsonr', 'pearson_residuals'}:
        warnings.warn(
            "svg(mode='pearsonr'/'pearson_residuals') selects highly variable "
            "genes from Pearson residuals and does not use spatial coordinates. "
            "Treat it as a non-spatial prefilter; use mode='moran', 'prost', "
            "'somde', or 'spatialde' for spatial evidence.",
            UserWarning,
            stacklevel=2,
        )
        from ..pp import preprocess
        bdata=adata.copy()
        # Pearson-residual HVG selection normalizes each spot by its total count,
        # so all-zero spots (common in Visium HD nucleus/cell segmentation, where
        # some segments capture no reads) trigger a `division by zero` (issue #860).
        # `bdata` is a throwaway copy used only to pick genes — its result is mapped
        # back to `adata` by gene name — so dropping empty spots here does not touch
        # the returned object's cells.
        import numpy as _np
        import scipy.sparse as _spp
        _counts = bdata.X
        _cell_sums = _np.asarray(
            _counts.sum(axis=1)).ravel() if _spp.issparse(_counts) else _np.asarray(_counts).sum(axis=1)
        _nonzero = _cell_sums > 0
        if not _nonzero.all():
            n_drop = int((~_nonzero).sum())
            print(f"   [WARN] Dropping {n_drop} all-zero spot(s) before Pearson-residual "
                  f"SVG selection (issue #860); returned AnnData is unchanged.")
            bdata = bdata[_nonzero].copy()
        bdata=preprocess(bdata,mode='shiftlog|pearson',n_HVGs=n_svgs,target_sum=target_sum)
        adata.var['space_variable_features'] = False
        both_genes = list(set(adata.var_names) & set(bdata.var_names))
        adata.var.loc[both_genes, 'space_variable_features'] = bdata.var.loc[both_genes, 'highly_variable']
        adata.uns.setdefault('space_svg_runs', {})['pearson_residuals'] = {
            'method': 'pearson_residuals',
            'spatial_evidence': False,
            'n_selected': int(adata.var['space_variable_features'].sum()),
            'compatibility_alias': mode == 'pearsonr',
        }
        add_reference(adata,'scanpy','non-spatial Pearson-residual HVG prefilter')
    elif mode in {'morani', 'moran'}:
        n_jobs        = kwargs.get('n_jobs', 1)
        n_perms       = kwargs.get('n_perms', 100)
        genes = adata.var_names.values
        spatial_neighbors(adata, library_key=library_key,
                          n_neighs=kwargs.get('n_neighs', 6), radius=kwargs.get('radius'))
        spatial_autocorr(
            adata,
            mode="moran",
            genes=genes,
            n_perms=n_perms,
            n_jobs=n_jobs,
            library_key=library_key,
            seed=kwargs.get('seed'), layer=kwargs.get('layer'),
        )
        adata.var['moranI'] = adata.uns['moranI']['I']
        pval_key = 'pval_sim' if n_perms is not None else 'pval_norm'
        adata.var['moranI_pval'] = adata.uns['moranI'][pval_key]
        adata.var['pval_adj'] = adata.uns['moranI']['pval_adj']
        #sort by moranI top 3000 genes
        adata.var['space_variable_features'] = False
        candidates = adata.var['moranI'].dropna()
        if selection == 'significant':
            candidates = candidates[(adata.var.loc[candidates.index, 'pval_adj'] < kwargs.get('qval_threshold', 0.05)) & (candidates > 0)]
        selected = candidates.sort_values(ascending=False).index if n_svgs is None else candidates.nlargest(n_svgs).index
        adata.var.loc[selected, 'space_variable_features'] = True
        add_reference(adata,'moranI','spatial variable gene selection with moranI')
    elif mode=='spateo':
        import spateo as st
        from ..pp import preprocess
        adata=preprocess(adata,mode='shiftlog|pearson',n_HVGs=n_svgs,target_sum=target_sum)
        e16_w, _ = st.svg.cal_wass_dis_bs(adata, **kwargs)
        # Add positive rate before smoothing for each gene
        st.svg.add_pos_ratio_to_adata(adata, layer='counts')
        e16_w['pos_ratio_raw'] = adata.var['pos_ratio_raw']
        # We obtain 529 significant SVGs
        sig_df = e16_w[(e16_w['log2fc']>=1) & (e16_w['rank_p']<=0.05) & (e16_w['pos_ratio_raw']>=0.05) & (e16_w['adj_pvalue']<=0.05)]
        adata.var['space_variable_features'] = False
        adata.var.loc[sig_df.index, 'space_variable_features'] = True
        print(f'{len(sig_df)} SVGs are selected!')
        print('In mode of spateo, the SVGs are selected based on the spatial expression pattern.')
        add_reference(adata,'spateo','spatial variable gene selection with spateo')
    elif mode == 'somde':
        import numpy as np
        import pandas as pd
        import scipy.sparse as _sp

        try:
            from ..external.somde import SomNode
        except ImportError as e:
            raise ImportError(
                "SOMDE requires `somoclu` and `patsy`. "
                "Install with: pip install somoclu patsy"
            ) from e

        k             = kwargs.get('k', 20)
        qval_thresh   = kwargs.get('qval_threshold', 0.05)
        retrain_epoch = kwargs.get('retrain_epoch', 0)
        n_jobs        = kwargs.get('n_jobs', 1)

        # --- spatial coordinates (cells × 2) ---
        X = adata.obsm['spatial'].astype(np.float32)

        # --- expression matrix (genes × cells) as DataFrame ---
        if 'counts' in adata.layers:
            mat = adata.layers['counts']
        else:
            mat = adata.X
        if _sp.issparse(mat):
            mat = mat.toarray()
        df = pd.DataFrame(
            mat.T,
            index=adata.var_names,
            columns=adata.obs_names,
        )

        print(f'Running SOMDE: {adata.n_obs} cells → ~{adata.n_obs // k} SOM nodes (k={k})')

        # --- SOM training ---
        som = SomNode(X, k)
        if retrain_epoch > 0:
            print(f'Re-training SOM for {retrain_epoch} additional epochs...')
            som.reTrain(retrain_epoch)

        # --- aggregate → normalize → SpatialDE test ---
        ndf, ninfo = som.mtx(df)
        nres = som.norm()
        result, SVnum = som.run(n_jobs=n_jobs)

        # --- store statistics back to adata.var ---
        result_indexed = result.set_index('g')
        result_indexed = result_indexed[~result_indexed.index.duplicated(keep='first')]
        for col in ('LLR', 'pval', 'qval', 'FSV'):
            if col in result_indexed.columns:
                adata.var[f'somde_{col}'] = result_indexed.reindex(adata.var_names)[col].values

        # --- select SVGs ---
        qvals = result_indexed.reindex(adata.var_names)['qval']
        adata.var['space_variable_features'] = False
        selected = _select_significant_svg_names(
            qvals,
            n_svgs=n_svgs,
            qval_threshold=qval_thresh if selection == 'significant' else None,
        )
        adata.var.loc[selected, 'space_variable_features'] = True

        add_reference(adata, 'SOMDE', 'spatial variable gene selection with SOMDE')
    elif mode == 'spatialde':
        import numpy as np
        import pandas as pd
        import scipy.sparse as _sp

        try:
            from ..external.SpatialDE import run as _spatialde_run
            from ..external.NaiveDE import stabilize as _stabilize, regress_out as _regress_out
        except ImportError as e:
            raise ImportError(
                "SpatialDE requires `patsy` and `tqdm`. "
                "Install with: pip install patsy tqdm"
            ) from e

        qval_thresh     = kwargs.get('qval_threshold', 0.05)
        min_total_count = kwargs.get('min_total_count', 3)
        regress_formula = kwargs.get('regress_formula', 'np.log(total_counts)')
        n_jobs          = kwargs.get('n_jobs', 1)
        kernel_space    = kwargs.get('kernel_space', None)
        approx_rank     = kwargs.get('approx_rank', None)
        approx_seed     = kwargs.get('approx_seed', 0)
        approx_models   = kwargs.get('approx_models', ('SE',))
        show_progress   = kwargs.get('show_progress', True)
        if isinstance(approx_models, str):
            approx_models = (approx_models,)

        # --- raw counts (cells × genes), avoid full densify before filtering ---
        if 'counts' in adata.layers:
            mat = adata.layers['counts']
        else:
            mat = adata.X

        if _sp.issparse(mat):
            gene_sum = np.asarray(mat.sum(axis=0)).ravel()
        else:
            mat = np.asarray(mat)
            gene_sum = mat.sum(axis=0)

        gene_mask = gene_sum >= min_total_count
        n_pass = int(gene_mask.sum())
        print(f'SpatialDE: {n_pass}/{adata.n_vars} genes pass '
              f'min_total_count={min_total_count} filter')

        # initialize outputs (keep full var length)
        adata.var['space_variable_features'] = False
        for col in ('LLR', 'pval', 'qval', 'FSV', 'l'):
            adata.var[f'spatialde_{col}'] = np.nan

        if n_pass == 0:
            print('No genes passed filtering; skipping SpatialDE run.')
            add_reference(adata, 'SpatialDE', 'spatial variable gene selection with SpatialDE')
            adata.var['highly_variable'] = adata.var['space_variable_features']
            return adata

        pass_idx = np.flatnonzero(gene_mask)
        pass_genes = np.asarray(adata.var_names)[pass_idx]

        if _sp.issparse(mat):
            counts_filt = mat[:, pass_idx].toarray()
        else:
            counts_filt = mat[:, pass_idx]
        counts_filt = np.asarray(counts_filt, dtype=np.float64, order='C')

        # --- sample_info: spatial coordinates + total_counts ---
        coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)
        sample_info = pd.DataFrame(coords, index=adata.obs_names, columns=['x', 'y'])
        sample_info['total_counts'] = counts_filt.sum(axis=1)

        # --- NaiveDE: variance stabilize → regress out library size ---
        # Shape convention for NaiveDE: genes × cells
        norm_expr_gxc = _stabilize(counts_filt.T)
        resid_expr_gxc = _regress_out(sample_info, norm_expr_gxc, regress_formula)
        resid_expr = pd.DataFrame(
            np.asarray(resid_expr_gxc, dtype=np.float64).T,
            index=adata.obs_names,
            columns=pass_genes,
        )

        # --- SpatialDE GP test ---
        # pass numpy array to avoid pandas multi-dim indexing deprecation in older spatialDE
        X_coords = sample_info[['x', 'y']].to_numpy(dtype=np.float64, copy=False)
        print(f'Running SpatialDE on {resid_expr.shape[1]} genes × '
              f'{resid_expr.shape[0]} cells (n_jobs={n_jobs})...')
        run_kwargs = {'n_jobs': n_jobs, 'use_tqdm': bool(show_progress)}
        if kernel_space is not None:
            run_kwargs['kernel_space'] = kernel_space
        if approx_rank is not None:
            if int(approx_rank) >= adata.n_obs:
                print(f'approx_rank={approx_rank} >= n_obs={adata.n_obs}; '
                      'falling back to exact eigendecomposition.')
            elif adata.n_obs <= 300:
                print('Warning: for small n_obs, Nyström may be slower than exact mode.')
            run_kwargs['approx_rank'] = int(approx_rank)
            run_kwargs['approx_seed'] = int(approx_seed)
            run_kwargs['approx_models'] = approx_models
            print(f'Using Nyström approximation: rank={approx_rank}, '
                  f'models={approx_models}, seed={approx_seed}')
        results = _spatialde_run(X_coords, resid_expr, **run_kwargs)

        # --- store per-gene statistics back to adata.var ---
        results_idx = results.set_index('g')
        results_idx = results_idx[~results_idx.index.duplicated(keep='first')]
        for col in ('LLR', 'pval', 'qval', 'FSV', 'l'):
            if col in results_idx.columns:
                adata.var.loc[pass_genes, f'spatialde_{col}'] = (
                    results_idx.reindex(pass_genes)[col].values
                )

        qvals = adata.var['spatialde_qval']
        adata.var['space_variable_features'] = False
        selected = _select_significant_svg_names(
            qvals,
            n_svgs=n_svgs,
            qval_threshold=qval_thresh if selection == 'significant' else None,
        )
        adata.var.loc[selected, 'space_variable_features'] = True

        add_reference(adata, 'SpatialDE', 'spatial variable gene selection with SpatialDE')
    else:
        raise ValueError(f"mode {mode} is not supported")

    adata.var['highly_variable'] = adata.var['space_variable_features']
    return adata
    # End-of-file (EOF)
