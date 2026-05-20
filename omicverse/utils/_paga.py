import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, issparse

from ._compat import PAGA
from .._registry import register_function

# TODO: Add docstrings
class PAGA_tree(PAGA):
    """TODO."""

    def __init__(
        self,
        adata,
        groups=None,
        vkey=None,
        use_time_prior=None,
        root_key=None,
        end_key=None,
        threshold_root_end_prior=None,
        minimum_spanning_tree=None,
    ):
        super().__init__(adata=adata, groups=groups, model="v1.2")
        self.groups = groups
        self.vkey = vkey
        self.use_time_prior = use_time_prior
        self.root_key = root_key
        self.end_key = end_key
        self.threshold_root_end_prior = threshold_root_end_prior
        if self.threshold_root_end_prior is None:
            self.threshold_root_end_prior = 0.9
        self.minimum_spanning_tree = minimum_spanning_tree

    # TODO: Add docstrings
    def compute_transitions(self):
        """TODO."""
        try:
            import igraph
        except ImportError:
            raise ImportError("To run paga, you need to install `pip install igraph`")
        vkey = f"{self.vkey}_graph"
        if vkey not in self._adata.uns:
            raise ValueError(
                "The passed AnnData needs to have an `uns` annotation "
                "with key 'velocity_graph' - a sparse matrix from RNA velocity."
            )
        if self._adata.uns[vkey].shape != (self._adata.n_obs, self._adata.n_obs):
            raise ValueError(
                f"The passed 'velocity_graph' has shape {self._adata.uns[vkey].shape} "
                f"but shoud have shape {(self._adata.n_obs, self._adata.n_obs)}"
            )

        clusters = self._adata.obs[self.groups]
        cats = clusters.cat.categories
        vgraph = self._adata.uns[vkey] > 0.1
        time_prior = self.use_time_prior

        if isinstance(time_prior, str) and time_prior in self._adata.obs.keys():
            vpt = self._adata.obs[time_prior].values
            vpt_mean = self._adata.obs.groupby(self.groups)[time_prior].mean()
            vpt_means = np.array([vpt_mean[cat] for cat in clusters])
            rows, cols, vals = [], [], []
            for i in range(vgraph.shape[0]):
                indices = vgraph[i].indices
                idx_bool = vpt[i] < vpt[indices]
                idx_bool &= vpt_means[indices] > vpt_means[i] - 0.1
                cols.extend(indices[idx_bool])
                vals.extend(vgraph[i].data[idx_bool])
                rows.extend([i] * np.sum(idx_bool))
            vgraph = vals_to_csr(vals, rows, cols, shape=vgraph.shape)

        lb = self.threshold_root_end_prior  # cells to be consider as terminal states
        if isinstance(self.end_key, str) and self.end_key in self._adata.obs.keys():
            set_row_csr(vgraph, rows=np.where(self._adata.obs[self.end_key] > lb)[0])
        if isinstance(self.root_key, str) and self.root_key in self._adata.obs.keys():
            vgraph[:, np.where(self._adata.obs[self.root_key] > lb)[0]] = 0
            vgraph.eliminate_zeros()

        membership = self._adata.obs[self.groups].cat.codes.values
        g = get_igraph_from_adjacency(vgraph, directed=True)
        vc = igraph.VertexClustering(g, membership=membership)
        cg_full = vc.cluster_graph(combine_edges="sum")
        transitions = get_sparse_from_igraph(cg_full, weight_attr="weight")
        transitions = transitions - transitions.T
        transitions_conf = transitions.copy()
        transitions = transitions.tocoo()
        total_n = self._neighbors.n_neighbors * np.array(vc.sizes())
        for i, j, v in zip(transitions.row, transitions.col, transitions.data):
            reference = np.sqrt(total_n[i] * total_n[j])
            transitions_conf[i, j] = 0 if v < 0 else v / reference
        transitions_conf.eliminate_zeros()

        # remove non-confident direct paths if more confident indirect path is found.
        T = transitions_conf.toarray()
        threshold = max(np.nanmin(np.nanmax(T / (T > 0), axis=0)) - 1e-6, 0.01)
        T *= T > threshold
        for i in range(len(T)):
            idx = T[i] > 0
            if np.any(idx):
                indirect = np.clip(T[idx], None, T[i][idx][:, None]).max(0)
                T[i, T[i] < indirect] = 0

        if self.minimum_spanning_tree:
            T_tmp = T.copy()
            T_num = T > 0
            T_sum = np.sum(T_num, 0)
            T_max = np.max(T_tmp)
            for i in range(len(T_tmp)):
                if T_sum[i] == 1:
                    T_tmp[np.where(T_num[:, i])[0][0], i] = T_max
            from scipy.sparse.csgraph import minimum_spanning_tree

            T_tmp = np.abs(minimum_spanning_tree(-T_tmp).toarray()) > 0
            T = T_tmp * T

        transitions_conf = csr_matrix(T)
        self.transitions_confidence = transitions_conf.T

        # set threshold for minimal spanning tree.
        df = pd.DataFrame(T, index=cats, columns=cats)
        self.threshold = np.nanmin(np.nanmax(df.values / (df.values > 0), axis=0))
        self.threshold = max(self.threshold - 1e-6, 0.01)


# TODO: Add docstrings
def vals_to_csr(vals, rows, cols, shape, split_negative=False):
    """TODO."""
    graph = coo_matrix((vals, (rows, cols)), shape=shape)

    if split_negative:
        graph_neg = graph.copy()

        graph.data = np.clip(graph.data, 0, 1)
        graph_neg.data = np.clip(graph_neg.data, -1, 0)

        graph.eliminate_zeros()
        graph_neg.eliminate_zeros()

        return graph.tocsr(), graph_neg.tocsr()

    else:
        return graph.tocsr()
    
# TODO: Finish docstrings
def set_row_csr(csr, rows, value=0):
    """Set all nonzero elements to the given value. Useful to set to 0 mostly."""
    for row in rows:
        start = csr.indptr[row]
        end = csr.indptr[row + 1]
        csr.data[start:end] = value
    if value == 0:
        csr.eliminate_zeros()


# TODO: Finish docstrings
def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shap[0] vertices
    g.add_edges(list(zip(sources, targets)))
    g.es["weight"] = weights
    if g.vcount() != adjacency.shape[0]:
        print(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes."
        )
    return g

# TODO: Add docstrings
def get_sparse_from_igraph(graph, weight_attr=None):
    """TODO."""
    edges = graph.get_edgelist()
    if weight_attr is None:
        weights = [1] * len(edges)
    else:
        # Convert to list to ensure proper extend behavior
        try:
            weights = list(graph.es[weight_attr])
        except (KeyError, AttributeError):
            # If weight attribute doesn't exist or is malformed, use default weights
            weights = [1] * len(edges)

    if not graph.is_directed():
        # Create a copy of the current weights before extending
        current_weights = weights.copy() if hasattr(weights, 'copy') else list(weights)
        edges.extend([(v, u) for u, v in edges])
        weights.extend(current_weights)

    shape = graph.vcount()
    shape = (shape, shape)

    # Check that we have valid edges and matching weights
    if len(edges) > 0 and len(weights) == len(edges):
        # Unpack edges into separate row and column index lists
        try:
            row_indices, col_indices = zip(*edges)
            # Convert to lists to ensure compatibility with scipy
            row_indices = list(row_indices)
            col_indices = list(col_indices)
            return csr_matrix((weights, (row_indices, col_indices)), shape=shape)
        except (ValueError, TypeError) as e:
            # If unpacking fails, return empty sparse matrix
            print(f"Warning: Failed to unpack edges: {e}. Returning empty sparse matrix.")
            return csr_matrix(shape)
    else:
        # Return empty sparse matrix with the correct shape
        return csr_matrix(shape)


@register_function(
    aliases=['计算 PAGA 图', 'cal_paga', 'trajectory graph abstraction'],
    category="utils",
    description="Compute PAGA connectivity among cell groups to summarize lineage topology and coarse-grained trajectory structure.",
    prerequisites={'functions': ['pp.neighbors'], 'optional_functions': ['pp.leiden']},
    requires={'obsp': ['connectivities'], 'uns': ['neighbors'], 'obs': ['cluster labels']},
    produces={'uns': ['paga'], 'obs': ['pseudotime (optional)']},
    auto_fix='escalate',
    examples=['ov.utils.cal_paga(adata, use_time_prior="dpt_pseudotime", vkey="paga")'],
    related=['utils.plot_paga', 'pp.neighbors', 'pp.umap']
)
def cal_paga(
    adata,
    groups=None,
    vkey="velocity",
    use_time_prior=True,
    root_key=None,
    end_key=None,
    threshold_root_end_prior=None,
    minimum_spanning_tree=True,
    copy=False,
):
    """Compute a PAGA graph with optional velocity/time priors.

    Parameters
    ----------
    adata:anndata.AnnData
        Annotated data matrix with neighborhood graph.
    groups:str or None
        Grouping key in ``adata.obs``; auto-detected when ``None``.
    vkey:str
        Velocity layer key used for transition confidence.
    use_time_prior:str or bool
        Pseudotime prior key (or ``True`` to auto-use velocity pseudotime).
    root_key:str or None
        Obs key marking root states.
    end_key:str or None
        Obs key marking terminal states.
    threshold_root_end_prior:float or None
        Threshold for root/end priors in ``[0, 1]``.
    minimum_spanning_tree:bool
        Whether to prune graph to most confident tree-like structure.
    copy:bool
        Whether to return a copied AnnData with computed results.

    Returns
    -------
    anndata.AnnData or None
        Copied AnnData if ``copy=True``; otherwise results are written in place.
    """
    if "neighbors" not in adata.uns:
        raise ValueError(
            "You need to run `pp.neighbors` first to compute a neighborhood graph."
        )

    adata = adata.copy() if copy else adata
    adata.uns['paga_graph']=adata.obsp['connectivities']
    strings_to_categoricals(adata)

    if groups is None:
        groups = (
            "clusters"
            if "clusters" in adata.obs.keys()
            else "louvain"
            if "louvain" in adata.obs.keys()
            else None
        )
    

    priors = [p for p in [use_time_prior, root_key, end_key] if p in adata.obs.keys()]
    print(
        "running PAGA",
        f"using priors: {priors}" if len(priors) > 0 else "",
    )
    paga = PAGA_tree(
        adata,
        groups,
        vkey=vkey,
        use_time_prior=use_time_prior,
        root_key=root_key,
        end_key=end_key,
        threshold_root_end_prior=threshold_root_end_prior,
        minimum_spanning_tree=minimum_spanning_tree,
    )

    if "paga" not in adata.uns:
        adata.uns["paga"] = {}

    paga.compute_connectivities()
    adata.uns["paga"]["connectivities"] = paga.connectivities
    adata.uns["paga"]["connectivities_tree"] = paga.connectivities_tree
    adata.uns[f"{groups}_sizes"] = np.array(paga.ns)

    paga.compute_transitions()
    adata.uns["paga"]["transitions_confidence"] = paga.transitions_confidence
    adata.uns["paga"]["threshold"] = paga.threshold
    adata.uns["paga"]["groups"] = groups

    print("    finished")
    print(
        "added\n" + "    'paga/connectivities', connectivities adjacency (adata.uns)\n"
        "    'paga/connectivities_tree', connectivities subtree (adata.uns)\n"
        "    'paga/transitions_confidence', velocity transitions (adata.uns)"
    )

    return adata if copy else None

def strings_to_categoricals(adata):
    """Transform string annotations to categoricals."""
    from pandas import Categorical
    from pandas.api.types import is_bool_dtype, is_integer_dtype, is_string_dtype

    def is_valid_dtype(values):
        return (
            is_string_dtype(values) or is_integer_dtype(values) or is_bool_dtype(values)
        )

    df = adata.obs
    df_keys = [key for key in df.columns if is_valid_dtype(df[key])]
    for key in df_keys:
        c = df[key]
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c

    df = adata.var
    df_keys = [key for key in df.columns if is_string_dtype(df[key])]
    for key in df_keys:
        c = df[key].astype("U")
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c

from pandas import Index

def make_unique_list(key, allow_array=False):
    """Normalize user input to a list of keys.

    Parameters
    ----------
    key:Any
        Input key or key collection.
    allow_array:bool
        Whether numpy arrays are treated as valid list-like inputs.

    Returns
    -------
    list or Any
        Normalized list-like key representation.
    """
    from collections import abc
    if isinstance(key, (Index, abc.KeysView)):
        key = list(key)
    is_list = (
        isinstance(key, (list, tuple, np.record))
        if allow_array
        else isinstance(key, (list, tuple, np.ndarray, np.record))
    )
    is_list_of_str = is_list and all(isinstance(item, str) for item in key)
    return key if is_list_of_str else key if is_list and len(key) < 20 else [key]

def check_basis(adata, basis):
    """Ensure embedding basis follows ``X_<basis>`` convention in ``obsm``.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData object containing embeddings.
    basis:str
        Embedding basis name.

    Returns
    -------
    None
        May add/rename ``adata.obsm`` key.
    """
    if basis in adata.obsm.keys() and f"X_{basis}" not in adata.obsm.keys():
        adata.obsm[f"X_{basis}"] = adata.obsm[basis]
        print(f"Renamed '{basis}' to convention 'X_{basis}' (adata.obsm).")

def make_unique_valid_list(adata, keys):
    """Filter key list to those available in AnnData containers.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData object with obs/var/layer keys.
    keys:list or str
        Candidate keys.

    Returns
    -------
    list
        Valid keys existing in AnnData.
    """
    keys = make_unique_list(keys)
    if all(isinstance(item, str) for item in keys):
        for i, key in enumerate(keys):
            if key.startswith("X_"):
                keys[i] = key = key[2:]
            check_basis(adata, key)
        valid_keys = np.hstack(
            [
                adata.obs.keys(),
                adata.var.keys(),
                adata.varm.keys(),
                adata.obsm.keys(),
                [key[2:] for key in adata.obsm.keys()],
                list(adata.layers.keys()),
            ]
        )
        keys_ = keys
        keys = [key for key in keys if key in valid_keys or key in adata.var_names]
        keys_ = [key for key in keys_ if key not in keys]
        if len(keys_) > 0:
            print(", ".join(keys_), "not found.")
    return keys

def default_basis(adata, **kwargs):
    """Choose a default embedding basis from AnnData.

    Parameters
    ----------
    adata:anndata.AnnData
        AnnData containing embeddings.
    **kwargs
        Optional explicit ``x``/``y`` vectors used to create temporary embedding.

    Returns
    -------
    str
        Selected basis name.
    """
    if "x" in kwargs and "y" in kwargs:
        keys, x, y = ["embedding"], kwargs.pop("x"), kwargs.pop("y")
        adata.obsm["X_embedding"] = np.stack([x, y]).T
        if "velocity_embedding" in adata.obsm.keys():
            del adata.obsm["velocity_embedding"]
    else:
        keys = [
            key for key in ["pca", "tsne", "umap"] if f"X_{key}" in adata.obsm.keys()
        ]
    if not keys:
        raise ValueError("No basis specified.")
    return keys[-1] if len(keys) > 0 else None

def _safe_categorical_to_str(adata, color_key):
    """
    Safely convert categorical values to string format for plotting.
    
    This function addresses pandas compatibility issues where direct assignment
    to cat.categories is no longer allowed in newer pandas versions.
    """
    if color_key in adata.obs.columns and hasattr(adata.obs[color_key], 'cat'):
        # Create a copy to avoid modifying the original data
        adata_copy = adata.copy()
        # Convert categorical values to string using cat.rename_categories
        if hasattr(adata_copy.obs[color_key].cat, 'categories'):
            str_categories = adata_copy.obs[color_key].cat.categories.astype(str)
            adata_copy.obs[color_key] = adata_copy.obs[color_key].cat.rename_categories(str_categories)
        return adata_copy
    return adata

@register_function(
    aliases=['绘制 PAGA 图', 'plot_paga', 'paga visualization'],
    category="utils",
    description="Plot PAGA graph over embedding coordinates to visualize lineage transitions and connectivity confidence.",
    prerequisites={'functions': ['cal_paga']},
    requires={'uns': ['paga'], 'obsm': ['X_umap or selected basis']},
    produces={},
    auto_fix='none',
    examples=['ov.utils.plot_paga(adata, basis="umap", title="PAGA graph")'],
    related=['utils.cal_paga', 'pl.embedding']
)
def plot_paga(adata,
    basis=None,
    vkey="velocity",
    color=None,
    layer=None,
    title=None,
    threshold=None,
    layout=None,
    layout_kwds=None,
    init_pos=None,
    root=0,
    labels=None,
    single_component=False,
    dashed_edges="connectivities",
    solid_edges="transitions_confidence",
    transitions="transitions_confidence",
    node_size_scale=1,
    node_size_power=0.5,
    edge_width_scale=0.4,
    min_edge_width=None,
    max_edge_width=2,
    arrowsize=15,
    random_state=0,
    pos=None,
    node_colors=None,
    normalize_to_color=False,
    cmap=None,
    cax=None,
    cb_kwds=None,
    add_pos=True,
    export_to_gexf=False,
    plot=True,
    use_raw=None,
    size=None,
    groups=None,
    components=None,
    figsize=None,
    dpi=None,
    show=None,
    save=None,
    ax=None,
    ncols=None,
    scatter_flag=None,
    **kwargs,):
    """
    Plot PAGA graph and optional embedding-level annotations.
    
    Parameters
    ----------
    adata:AnnData
        AnnData with PAGA graph stored in ``adata.uns['paga']``.
    basis:str or None, optional
        Embedding basis used for overlay (for example ``'umap'``). Set ``None`` to draw
        graph layout only.
    vkey:str, optional
        Velocity key used by scVelo when velocity-aware plotting is enabled.
    color:str or sequence of str, optional
        Cell-level annotation(s) used for coloring.
    layer:str or None, optional
        Layer used for expression coloring.
    title:str or None, optional
        Plot title.
    threshold:float or None, optional
        Minimum edge threshold.
    layout:str or None, optional
        Graph layout algorithm name.
    layout_kwds:dict or None, optional
        Additional layout options.
    init_pos:str or None, optional
        Initialization for graph layout.
    root:int or None, optional
        Root index used by tree-like layouts.
    labels:str or None, optional
        Node labels to display.
    single_component:bool, optional
        Keep only the largest connected component.
    dashed_edges:str or None, optional
        Key for dashed edges.
    solid_edges:str or None, optional
        Key for solid edges.
    transitions:str or None, optional
        Transition-confidence edge key.
    node_size_scale:float, optional
        Scale factor for node sizes.
    node_size_power:float, optional
        Exponent for node-size scaling.
    edge_width_scale:float, optional
        Scale factor for edge widths.
    min_edge_width:float or None, optional
        Lower bound for edge widths.
    max_edge_width:float, optional
        Upper bound for edge widths.
    arrowsize:float, optional
        Arrow size for directed edges.
    random_state:int, optional
        Random seed used by layout routines.
    pos:dict or None, optional
        Precomputed node positions.
    node_colors:dict or sequence, optional
        Custom node colors.
    normalize_to_color:bool, optional
        Whether to normalize values before color mapping.
    cmap:str or matplotlib colormap, optional
        Colormap.
    cax:matplotlib.axes.Axes or None, optional
        Colorbar axis.
    cb_kwds:dict or None, optional
        Colorbar keyword arguments.
    add_pos:bool, optional
        Whether to save computed positions back to ``adata.uns``.
    export_to_gexf:bool, optional
        Whether to export graph to GEXF.
    plot:bool, optional
        If ``False``, only compute/return graph data.
    use_raw:bool or None, optional
        Whether to use ``adata.raw``.
    size:float or None, optional
        Point size for scatter overlay.
    groups:str or sequence or None, optional
        Category subset.
    components:str or sequence or None, optional
        Embedding components to plot.
    figsize:tuple or None, optional
        Figure size.
    dpi:int or None, optional
        Figure dpi.
    show:bool or None, optional
        Whether to immediately show the figure.
    save:str or bool or None, optional
        Save path or Scanpy-style save flag.
    ax:matplotlib.axes.Axes or None, optional
        Existing axis.
    ncols:int or None, optional
        Number of columns when plotting multiple panels.
    scatter_flag:bool or None, optional
        Whether to add embedding scatter underlay.
    **kwargs
        Additional arguments forwarded to ``scvelo.pl.paga``.
    
    Returns
    -------
    Any
        Return value from ``scvelo.pl.paga``.
    
    Examples
    --------
    >>> ov.utils.plot_paga(adata, basis='umap', color='celltype', threshold=0.05)
    """

    if layout is not None:
        basis = None
    
    # Fix for pandas compatibility: when using basis with categorical colors,
    # scvelo may fail due to read-only categories in newer pandas versions
    adata_processed = adata
    if basis is not None and color is not None:
        adata_processed = _safe_categorical_to_str(adata, color)
    
    import scvelo as scv
    return scv.pl.paga(adata_processed,
    basis=basis,
    vkey=vkey,
    color=color,
    layer=layer,
    title=title,
    threshold=threshold,
    layout=layout,
    layout_kwds=layout_kwds,
    init_pos=init_pos,
    root=root,
    labels=labels,
    single_component=single_component,
    dashed_edges=dashed_edges,
    solid_edges=solid_edges,
    transitions=transitions,
    node_size_scale=node_size_scale,
    node_size_power=node_size_power,
    edge_width_scale=edge_width_scale,
    min_edge_width=min_edge_width,
    max_edge_width=max_edge_width,
    arrowsize=arrowsize,
    random_state=random_state,
    pos=pos,
    node_colors=node_colors,
    normalize_to_color=normalize_to_color,
    cmap=cmap,
    cax=cax,
    cb_kwds=cb_kwds,
    add_pos=add_pos,
    export_to_gexf=export_to_gexf,
    plot=plot,
    use_raw=use_raw,
    size=size,
    groups=groups,
    components=components,
    figsize=figsize,
    dpi=dpi,
    show=show,
    save=save,
    ax=ax,
    ncols=ncols,
    scatter_flag=scatter_flag,
    **kwargs,)
