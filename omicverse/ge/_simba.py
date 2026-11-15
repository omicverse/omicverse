"""Native SIMBA (single-cell embedding into multi-modal space) on ``ov.ge``.

A from-scratch port of `pinellolab/simba <https://github.com/pinellolab/simba>`_
+ its vendored `simba_pbg <https://github.com/pinellolab/simba_pbg>`_ (a fork
of PyTorch-BigGraph): no ``import simba`` / ``import torchbiggraph`` anywhere.
Preprocessing (``normalize``/``log_transform``/``discretize``/
``select_variable_genes``) and graph construction (``gen_graph``,
``infer_edges``) reproduce the reference's exact math (see each
function's docstring for the source it mirrors); training runs on
:class:`~omicverse.ge._base.BaseGraphModel` instead of PBG's Hogwild
multi-process trainer (see ``SPEC_ge_simba.md`` for what's intentionally
different and why).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._registry import register_function
from ._base import BaseGraphModel
from ._graph import HeteroGraph

# ----------------------------------------------------------------------
# preprocessing (simba/preprocessing/_general.py, tools/_general.py)
# ----------------------------------------------------------------------


@register_function(
    aliases=["ge.log_transform", "simba log transform"],
    category="ge",
    description="log1p-transform a (count) AnnData matrix, SIMBA's preprocessing convention.",
    examples=["ov.ge.log_transform(adata)"],
    related=["ge.normalize", "ge.discretize"],
)
def log_transform(adata, layer=None):
    """``np.log1p`` of the (count) matrix (``simba.pp.log_transform``)."""
    X = adata.X if layer is None else adata.layers[layer]
    X = sp.csr_matrix(X) if not sp.issparse(X) else X
    out = X.copy()
    out.data = np.log1p(out.data)
    if layer is None:
        adata.X = out
    else:
        adata.layers[layer] = out
    return adata


def binarize(adata, threshold: float = 1e-5, layer=None):
    """Zero out entries ``<= threshold``, set the rest to 1 (``simba.pp.binarize``)."""
    X = adata.X if layer is None else adata.layers[layer]
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.copy()
    X = X.copy()
    X.data = (X.data > threshold).astype(X.dtype)
    X.eliminate_zeros()
    if layer is None:
        adata.X = X
    else:
        adata.layers[layer] = X
    return adata


def _tf_idf(mat: sp.csr_matrix) -> sp.csr_matrix:
    """``simba.preprocessing._utils.cal_tf_idf``: TF-IDF with log-scaled IDF."""
    mat = sp.csr_matrix(mat, dtype=np.float64)
    n_docs = mat.shape[0]
    tf = mat.multiply(1.0 / (np.asarray(mat.sum(axis=1)).flatten()[:, None] + 1e-12))
    df = np.asarray((mat > 0).sum(axis=0)).flatten()
    idf = np.log(1 + n_docs / (df + 1e-12))
    return sp.csr_matrix(tf).multiply(idf).tocsr()


@register_function(
    aliases=["ge.normalize", "simba normalize"],
    category="ge",
    description="Library-size or TF-IDF normalisation, SIMBA's preprocessing convention.",
    examples=["ov.ge.normalize(adata, method='lib_size')"],
    related=["ge.log_transform", "ge.discretize"],
)
def normalize(adata, method: str = "lib_size", scale_factor: float = 1e4,
              save_raw: bool = True, layer=None):
    """Library-size (CPM-like) or TF-IDF normalisation (``simba.pp.normalize``)."""
    if method not in ("lib_size", "tf_idf"):
        raise ValueError(f"unrecognized method {method!r}")
    X = adata.X if layer is None else adata.layers[layer]
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.copy()
    if save_raw:
        adata.layers["raw"] = X.copy()
    if method == "lib_size":
        row_sum = np.asarray(X.sum(axis=1)).flatten()
        row_sum[row_sum == 0] = 1.0
        from sklearn.utils import sparsefuncs
        sparsefuncs.inplace_row_scale(X, 1.0 / row_sum)
        X = X * scale_factor
    else:
        X = _tf_idf(X)
    if layer is None:
        adata.X = X
    else:
        adata.layers[layer] = X
    return adata


@register_function(
    aliases=["ge.select_variable_genes", "simba highly variable genes"],
    category="ge",
    description="Seurat v3 vst highly-variable-gene selection, SIMBA's preprocessing convention.",
    examples=["ov.ge.select_variable_genes(adata, n_top_genes=2000)"],
    related=["pp.highly_variable_genes", "ge.discretize"],
)
def select_variable_genes(adata, layer: str = "raw", n_top_genes: int = 2000,
                          span: float = 0.3):
    """Seurat v3 ``vst`` highly-variable-gene selection (``simba.pp.select_variable_genes``).

    Delegates to :func:`omicverse.pp.highly_variable_genes` (``flavor=
    'seurat_v3'``), the same loess-regularised-variance algorithm SIMBA's own
    implementation uses, and copies the result into the column names SIMBA
    expects (``variances_norm``/``variances``/``means``/``highly_variable``).
    """
    from ..pp import highly_variable_genes

    df = highly_variable_genes(adata, layer=layer, flavor="seurat_v3", n_top_genes=n_top_genes,
                               span=span, inplace=False, check_values=False)
    adata.var["variances_norm"] = df["variances_norm"].to_numpy()
    adata.var["variances"] = df["variances"].to_numpy()
    adata.var["means"] = df["means"].to_numpy()
    adata.var["highly_variable"] = df["highly_variable"].to_numpy()
    return adata


@register_function(
    aliases=["ge.discretize", "simba discretize", "表达离散化"],
    category="ge",
    description="Discretize non-zero expression into n_bins levels by weighted 1-D k-means (SIMBA).",
    produces={"layers": ["disc"], "uns": ["disc"]},
    examples=["ov.ge.discretize(adata, n_bins=5)"],
    related=["ge.gen_graph", "ge.SIMBA"],
)
def discretize(adata, layer=None, n_bins: int = 5, max_bins: int = 100, seed: int = 2021):
    """Discretize non-zero continuous values into ``n_bins`` levels by
    1-D k-means on a histogram approximation (``simba.tl.discretize``,
    ported verbatim including the histogram-weighted ``KMeans`` and the
    padding added to the outer bin edges)."""
    from sklearn.cluster import KMeans

    X = adata.X if layer is None else adata.layers[layer]
    X = sp.csr_matrix(X) if not sp.issparse(X) else X
    nonzero = X.data

    hist_count, hist_edges = np.histogram(nonzero, bins=max_bins, density=False)
    hist_centroids = (hist_edges[:-1] + hist_edges[1:]) / 2

    km = KMeans(n_clusters=n_bins, random_state=seed, n_init=10).fit(
        hist_centroids.reshape(-1, 1), sample_weight=hist_count)
    cluster_centers = np.sort(km.cluster_centers_.flatten())

    padding = (hist_edges[-1] - hist_edges[0]) / (max_bins * 10)
    bin_edges = np.array([hist_edges[0] - padding]
                         + list((cluster_centers[:-1] + cluster_centers[1:]) / 2)
                         + [hist_edges[-1] + padding])
    nonzero_disc = np.digitize(nonzero, bin_edges).reshape(-1)
    bin_count = np.unique(nonzero_disc, return_counts=True)[1]

    disc = X.copy()
    disc.data = nonzero_disc.astype(np.float64)
    adata.layers["disc"] = disc
    adata.uns["disc"] = {"bin_edges": bin_edges, "bin_count": bin_count,
                         "hist_edges": hist_edges, "hist_count": hist_count}
    return adata


# ----------------------------------------------------------------------
# graph construction (simba/tools/_pbg.py:gen_graph, _integration.py)
# ----------------------------------------------------------------------


@register_function(
    aliases=["ge.gen_graph", "simba gen_graph"],
    category="ge",
    description="Build a HeteroGraph from SIMBA's CP/PM/PK/CG/CC AnnData recipes.",
    examples=["g = ov.ge.gen_graph(list_CG=[adata])"],
    related=["ge.SIMBA", "ge.infer_edges", "ge.HeteroGraph"],
)
def gen_graph(list_CG=None, list_CC=None, list_CP=None, list_PM=None, list_PK=None, *,
              prefix_C: str = "C", prefix_G: str = "G", prefix_P: str = "P",
              prefix_M: str = "M", prefix_K: str = "K",
              use_highly_variable: bool = True, use_top_pcs: bool = True,
              use_top_pcs_CP=None, use_top_pcs_PM=None, use_top_pcs_PK=None) -> HeteroGraph:
    """Build a :class:`~omicverse.ge._graph.HeteroGraph` from the same
    (CP, PM, PK, CG, CC) AnnData "recipes" as ``simba.tl.gen_graph``: entity
    ids, relation weights (``1.0`` CP / ``0.2`` PM / ``0.02`` PK /
    ``linspace(1, 5)`` per CG expression level / ``10.0`` CC) and the
    automatic splitting of cell-containing AnnData into distinct cell
    *entity types* (``C``, ``C2``, ``C3``, ...) whenever one dataset's
    cells are not a subset of any already-registered cell set -- ported
    from ``simba/tools/_pbg.py``.
    """
    if all(x is None for x in (list_CP, list_PM, list_PK, list_CG, list_CC)):
        raise ValueError("at least one of list_CP/list_PM/list_PK/list_CG/list_CC must be given")

    graph = HeteroGraph()
    dict_cells: dict = {}

    def cell_key(obs_names) -> str:
        obs_names = pd.Index(obs_names)
        if not dict_cells:
            dict_cells[prefix_C] = obs_names
            return prefix_C
        obs_set = set(obs_names)
        for key, ids in dict_cells.items():
            if obs_set <= set(ids):
                return key
        key = f"{prefix_C}{len(dict_cells) + 1}"
        dict_cells[key] = obs_names
        return key

    if list_CP is not None:
        for adata_ori in list_CP:
            flag = use_top_pcs_CP if use_top_pcs_CP is not None else use_top_pcs
            adata = adata_ori[:, adata_ori.var["top_pcs"]] if flag else adata_ori
            key = cell_key(adata.obs_names)
            graph.add_edges_from_sparse(key, prefix_P, adata.X, adata.obs_names,
                                        adata.var_names, weight=1.0)

    if list_PM is not None:
        for adata_ori in list_PM:
            flag = use_top_pcs_PM if use_top_pcs_PM is not None else use_top_pcs
            adata = adata_ori[:, adata_ori.var["top_pcs"]] if flag else adata_ori
            graph.add_edges_from_sparse(prefix_P, prefix_M, adata.X, adata.obs_names,
                                        adata.var_names, weight=0.2)

    if list_PK is not None:
        for adata_ori in list_PK:
            flag = use_top_pcs_PK if use_top_pcs_PK is not None else use_top_pcs
            adata = adata_ori[:, adata_ori.var["top_pcs"]] if flag else adata_ori
            graph.add_edges_from_sparse(prefix_P, prefix_K, adata.X, adata.obs_names,
                                        adata.var_names, weight=0.02)

    if list_CG is not None:
        for adata_ori in list_CG:
            adata = (adata_ori[:, adata_ori.var["highly_variable"]]
                    if use_highly_variable else adata_ori)
            key = cell_key(adata.obs_names)
            graph.add_discretized_relations(key, prefix_G, adata.layers["disc"],
                                            adata.obs_names, adata.var_names,
                                            weight_min=1.0, weight_max=5.0)

    if list_CC is not None:
        for adata in list_CC:
            key_obs = cell_key(adata.obs_names)
            key_var = cell_key(adata.var_names)
            graph.add_edges_from_sparse(key_obs, key_var, adata.layers["conn"],
                                        adata.obs_names, adata.var_names, weight=10.0)

    return graph


def _knn_conn_dist(X_ref, X_query, k, leaf_size, metric):
    """``simba.tools._utils._knn``: for every row of ``X_query``, the ``k``
    nearest rows of ``X_ref``; returns ``(n_ref, n_query)`` connectivity /
    distance matrices."""
    from sklearn.neighbors import KDTree

    kdt = KDTree(X_ref, leaf_size=leaf_size, metric=metric)
    dist, ind = kdt.query(X_query, k=k, return_distance=True)
    row = np.repeat(np.arange(ind.shape[0]), ind.shape[1])
    col = ind.flatten()
    conn = sp.csr_matrix((np.ones(len(row)), (row, col)),
                         shape=(X_query.shape[0], X_ref.shape[0])).T.tocsr()
    dmat = sp.csr_matrix((dist.flatten(), (row, col)),
                         shape=(X_query.shape[0], X_ref.shape[0])).T.tocsr()
    return conn, dmat


@register_function(
    aliases=["ge.infer_edges", "simba infer_edges"],
    category="ge",
    description="Mutual-nearest-neighbour edges between reference and query cells (SIMBA integration).",
    examples=["adata_cc = ov.ge.infer_edges(adata_ref, adata_query)"],
    related=["ge.gen_graph", "ge.trim_edges", "single.pySIMBA"],
)
def infer_edges(adata_ref, adata_query, *, feature: str = "highly_variable",
                n_components: int = 20, random_state: int = 42, layer=None,
                k: int = 20, metric: str = "euclidean", leaf_size: int = 40, **kwargs):
    """Mutual-nearest-neighbour edges between reference and query cells
    (``simba.tl.infer_edges``): the shared ``feature`` columns are projected
    through a truncated SVD of ``X_ref @ X_query.T``, then (ref, query)
    pairs that are each in the other's top-``k`` neighbours in that shared
    space become edges, weighted by inverse distance + 1.

    Returns an ``AnnData`` (obs = ref cells, var = query cells) with the
    binary mutual-kNN mask in ``.layers['conn']`` and the similarity scores
    in ``.X`` -- the input :func:`gen_graph`'s ``list_CC`` expects.
    """
    import anndata as ad
    from sklearn.utils.extmath import randomized_svd

    mask_ref = adata_ref.var[feature].to_numpy()
    feature_ref = set(adata_ref.var_names[mask_ref])
    feature_shared = list(feature_ref & set(adata_query.var_names))
    print(f"#shared features: {len(feature_shared)}")

    X_ref = (adata_ref[:, feature_shared].X if layer is None
            else adata_ref[:, feature_shared].layers[layer])
    X_query = (adata_query[:, feature_shared].X if layer is None
              else adata_query[:, feature_shared].layers[layer])
    X_ref = sp.csr_matrix(X_ref)
    X_query = sp.csr_matrix(X_query)
    if (np.asarray(X_ref.sum(axis=1)).flatten() == 0).any() or \
       (np.asarray(X_query.sum(axis=1)).flatten() == 0).any():
        raise ValueError(f"some cells have zero expressed {feature!r} features; "
                         f"try including more {feature} features")

    print("Performing randomized SVD ...")
    mat = X_ref @ X_query.T
    U, _, VT = randomized_svd(mat, n_components=n_components, random_state=random_state,
                              **kwargs)
    svd_data = np.vstack((U, VT.T))
    X_svd_ref = svd_data[:U.shape[0], :]
    X_svd_query = svd_data[-VT.shape[1]:, :]
    X_svd_ref = X_svd_ref / np.sqrt((X_svd_ref ** 2).sum(-1, keepdims=True))
    X_svd_query = X_svd_query / np.sqrt((X_svd_query ** 2).sum(-1, keepdims=True))

    print("Searching for mutual nearest neighbors ...")
    knn_conn_rq, knn_dist_rq = _knn_conn_dist(X_svd_ref, X_svd_query, k, leaf_size, metric)
    knn_conn_qr, _ = _knn_conn_dist(X_svd_query, X_svd_ref, k, leaf_size, metric)

    sum_conn = knn_conn_rq + knn_conn_qr.T
    id_x, id_y, values = sp.find(sum_conn > 1)
    print(f"{len(id_x)} edges are selected")
    conn_ref_query = sp.csr_matrix((values * 1, (id_x, id_y)), shape=sum_conn.shape)
    dist_sel = np.asarray(knn_dist_rq[id_x, id_y]).flatten()
    dist_ref_query = sp.csr_matrix((dist_sel, (id_x, id_y)), shape=sum_conn.shape)
    sim_ref_query = sp.csr_matrix(
        (1 / (dist_ref_query.data + 1), dist_ref_query.nonzero()), shape=dist_ref_query.shape)

    adata_ref_query = ad.AnnData(X=sim_ref_query, obs=adata_ref.obs.copy(),
                                 var=adata_query.obs.copy())
    adata_ref_query.layers["conn"] = conn_ref_query
    adata_ref_query.obsm["svd"] = X_svd_ref
    adata_ref_query.varm["svd"] = X_svd_query
    return adata_ref_query


def trim_edges(adata_ref_query, cutoff=None, n_edges=None):
    """Keep only similarity scores above ``cutoff`` (or the top ``n_edges``)
    (``simba.tl.trim_edges``)."""
    sim = adata_ref_query.X
    if cutoff is None:
        if n_edges is None:
            raise ValueError('"cutoff" or "n_edges" must be given')
        cutoff = np.partition(sim.data, sim.size - n_edges)[sim.size - n_edges]
    id_x, id_y, values = sp.find(sim > cutoff)
    print(f"{len(id_x)} edges are selected")
    adata_ref_query.layers["conn"] = sp.csr_matrix((values * 1, (id_x, id_y)), shape=sim.shape)
    return adata_ref_query


# ----------------------------------------------------------------------
# post-training analysis (simba/tools/_post_training.py)
# ----------------------------------------------------------------------


def softmax_transform(X_ref: np.ndarray, X_query: np.ndarray, T: float = 0.5,
                      n_top=None, percentile: float = 0) -> np.ndarray:
    """Project ``X_query`` embeddings onto the ``X_ref`` reference space by
    a softmax-weighted average of reference entities (``simba.tl.softmax``)."""
    scores = X_ref @ X_query.T
    scores = scores - scores.max()
    probs = np.exp(scores / T) / np.exp(scores / T).sum(axis=0, keepdims=True)
    if n_top is None:
        thresh = np.percentile(probs, q=percentile, axis=0)
    else:
        thresh = np.sort(probs, axis=0)[::-1, :][n_top - 1, :]
    probs = np.where(probs < thresh[None, :], 0.0, probs)
    probs = probs / probs.sum(axis=0, keepdims=True)
    return probs.T @ X_ref


def embed(adata_ref, list_adata_query: list, *, T: float = 0.5, list_T=None,
          percentile: float = 0, n_top=None, list_percentile=None):
    """Merge a reference embedding with one or more query embeddings into a
    single comparable space via :func:`softmax_transform`
    (``simba.tl.embed``)."""
    import anndata as ad

    X_all = adata_ref.X.copy()
    obs_all = adata_ref.obs.copy()
    obs_all["id_dataset"] = "ref"
    frames = [obs_all]
    for i, adata_query in enumerate(list_adata_query):
        T_i = (list_T[i] if list_T is not None else T)
        pct_i = (list_percentile[i] if list_percentile is not None else percentile)
        X_proj = softmax_transform(adata_ref.X, adata_query.X, T=T_i, n_top=n_top,
                                   percentile=pct_i)
        adata_query.layers["softmax"] = X_proj
        X_all = np.vstack((X_all, X_proj))
        obs_query = adata_query.obs.copy()
        obs_query["id_dataset"] = f"query_{i}"
        frames.append(obs_query)
    obs_all = pd.concat(frames, axis=0)
    return ad.AnnData(X=X_all, obs=obs_all)


def compare_entities(adata_ref, adata_query, n_top_cells: int = 50, T: float = 1.0):
    """Per-query-entity SIMBA metrics against a reference embedding --
    ``max``/``std``/``gini``/``entropy`` (``simba.tl.compare_entities``)."""
    import anndata as ad
    from scipy.stats import entropy

    from ._utils import gini as _gini

    X_ref, X_query = adata_ref.X, adata_query.X
    X_cmp = X_ref @ X_query.T
    adata_cmp = ad.AnnData(X=X_cmp, obs=adata_ref.obs.copy(), var=adata_query.obs.copy())
    adata_cmp.layers["norm"] = X_cmp - np.log(np.exp(X_cmp).mean(axis=0)).reshape(1, -1)
    adata_cmp.layers["softmax"] = np.exp(X_cmp / T) / np.exp(X_cmp / T).sum(axis=0).reshape(1, -1)
    adata_cmp.var["max"] = np.clip(
        np.sort(adata_cmp.layers["norm"], axis=0)[-n_top_cells:, ], a_min=0, a_max=None
    ).mean(axis=0)
    adata_cmp.var["std"] = np.std(X_cmp, axis=0, ddof=1)
    adata_cmp.var["gini"] = np.array([_gini(adata_cmp.layers["softmax"][:, i])
                                      for i in range(X_cmp.shape[1])])
    adata_cmp.var["entropy"] = entropy(adata_cmp.layers["softmax"])
    return adata_cmp


def query(adata, *, obsm: str = "X_simba", entity=None, k: int = 20, metric: str = "euclidean"):
    """``k``-nearest entities (by embedding distance) to ``entity``
    (a simplified, embedding-only ``simba.tl.query``)."""
    from sklearn.neighbors import KDTree

    X = adata.obsm[obsm] if obsm in getattr(adata, "obsm", {}) else adata.X
    pin = np.atleast_2d(adata[entity, :].obsm[obsm] if obsm in getattr(adata, "obsm", {})
                        else adata[entity, :].X)
    kdt = KDTree(X, metric=metric)
    dist, ind = kdt.query(pin, k=k, return_distance=True)
    frames = []
    for i in range(pin.shape[0]):
        df = adata.obs.iloc[ind[i]].copy()
        df["distance"] = dist[i]
        df["query"] = np.atleast_1d(entity)[i]
        frames.append(df)
    out = pd.concat(frames, axis=0).sort_values("distance")
    return out


# ----------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------


@register_function(
    aliases=["ge.SIMBA", "native SIMBA", "SIMBA图嵌入"],
    category="ge",
    description="Native SIMBA knowledge-graph embedding of cells/genes (optionally peaks/motifs/k-mers), pure PyTorch (no simba/simba_pbg install).",
    prerequisites={"optional_functions": ["ge.normalize", "ge.discretize", "ge.select_variable_genes"]},
    produces={"obsm": ["X_simba embedding via get_embedding"]},
    examples=["m = ov.ge.SIMBA(dimension=50); m.add_cg(adata); m.train(num_epochs=10)"],
    related=["single.pySIMBA", "ge.gen_graph", "ge.BaseGraphModel"],
)
class SIMBA(BaseGraphModel):
    """Native SIMBA: PBG-style knowledge-graph embedding of cells, genes,
    (and optionally peaks/motifs/k-mers) specialised with SIMBA's own graph
    recipe and preprocessing. See :class:`~omicverse.ge._base.BaseGraphModel`
    for training/persistence and ``ov.single.pySIMBA`` for the batch
    integration workflow that drives this class end to end.
    """

    _model_name = "SIMBA"

    def add_cg(self, adata, *, cell_type: str = "C", gene_type: str = "G",
              use_highly_variable: bool = True, disc_layer: str = "disc",
              weight_min: float = 1.0, weight_max: float = 5.0) -> "SIMBA":
        """Register one cell-gene (discretized expression) relation set --
        one relation per expression level, as in :func:`gen_graph`."""
        sub = (adata[:, adata.var["highly_variable"]]
              if (use_highly_variable and "highly_variable" in adata.var) else adata)
        self.graph.add_discretized_relations(cell_type, gene_type, sub.layers[disc_layer],
                                             sub.obs_names, sub.var_names,
                                             weight_min=weight_min, weight_max=weight_max)
        return self

    def add_cc(self, adata_ref_query, *, lhs_type: str, rhs_type: str, weight: float = 10.0,
              conn_layer: str = "conn") -> "SIMBA":
        """Register a cell-cell (mutual-kNN, e.g. from :func:`infer_edges`)
        relation between two (already-registered) cell entity types."""
        self.graph.add_edges_from_sparse(lhs_type, rhs_type, adata_ref_query.layers[conn_layer],
                                         adata_ref_query.obs_names, adata_ref_query.var_names,
                                         weight=weight)
        return self

    def embed_query(self, ref_type: str, query_types, **kwargs):
        """Merge ``ref_type``'s embedding with one or more other cell type
        embeddings into one comparable space (:func:`embed`)."""
        query_types = [query_types] if isinstance(query_types, str) else list(query_types)
        return embed(self.get_embedding(ref_type), [self.get_embedding(t) for t in query_types],
                    **kwargs)

    def compare_entities(self, ref_type: str, query_type: str, **kwargs):
        return compare_entities(self.get_embedding(ref_type), self.get_embedding(query_type),
                                **kwargs)


__all__ = ["SIMBA", "gen_graph", "infer_edges", "trim_edges", "discretize", "normalize",
           "log_transform", "binarize", "select_variable_genes", "softmax_transform",
           "embed", "compare_entities", "query"]
