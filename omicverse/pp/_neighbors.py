"""Functions and classes for computing nearest neighbors."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from textwrap import indent
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, TypedDict
from warnings import warn

import numpy as np
import scipy
from scipy import sparse
from sklearn.utils import check_random_state


SpBase = sparse.spmatrix | sparse.sparray  # noqa: TID251
"""Only use when you directly convert it to a known subclass."""

_CSArray = sparse.csr_array | sparse.csc_array  # noqa: TID251
"""Only use if you want to specially handle arrays as opposed to matrices."""

_CSMatrix = sparse.csr_matrix | sparse.csc_matrix  # noqa: TID251
"""Only use if you want to specially handle matrices as opposed to arrays."""

CSRBase = sparse.csr_matrix | sparse.csr_array  # noqa: TID251
CSCBase = sparse.csc_matrix | sparse.csc_array  # noqa: TID251
CSBase = _CSArray | _CSMatrix

from typing import TYPE_CHECKING, Literal, Protocol
import logging

from ._compat import old_positionals
from .._settings import EMOJI, Colors, settings as ov_settings
from .._optional import normalize_torch_device
from . import _connectivity
from ._connectivity import (
    _get_indices_distances_from_sparse_matrix,
    _get_sparse_matrix_from_indices_distances,
)

# Check torch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Create logger
logg = logging.getLogger(__name__)

# Type definitions
_Method = Literal["umap", "gauss", "torch"]
_KnownTransformer = Literal["pynndescent", "rapids", "sklearn", "pyg"]
_Metric = Literal[
    "euclidean", "manhattan", "chebyshev", "minkowski", "canberra",
    "braycurtis", "haversine", "mahalanobis", "wminkowski", "seuclidean",
    "cosine", "correlation"
]

# Doc strings for parameters
doc_n_pcs = """\
n_pcs
    Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`."""

doc_use_rep = """\
use_rep
    Use the indicated representation. `'X'` or any key for `.obsm` is valid.
    If `None`, the representation is chosen automatically:
    For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
    If 'X_pca' is not present, it's computed with default parameters."""


# ================================================================================
# Utility functions from scanpy (to remove scanpy dependency)
# ================================================================================

def _doc_params(**replacements: str):
    """Decorator to replace parameters in docstrings."""
    def dec(obj):
        if obj.__doc__:
            obj.__doc__ = obj.__doc__.format(**replacements)
        return obj
    return dec


def get_literal_vals(literal_type):
    """Get values from a Literal type."""
    import typing
    if hasattr(typing, 'get_args'):
        return typing.get_args(literal_type)
    else:
        # Fallback for older Python versions
        return literal_type.__args__ if hasattr(literal_type, '__args__') else ()


class NeighborsView:
    """View into neighbors data in AnnData object."""

    def __init__(self, adata, key: str = "neighbors"):
        self._adata = adata
        self._key = key
        if key not in adata.uns:
            msg = f"No {key!r} in .uns"
            raise KeyError(msg)
        self._neighbors_dict = adata.uns[key]

        # Get keys for connectivities and distances
        if "connectivities_key" in self._neighbors_dict:
            self._conns_key = self._neighbors_dict["connectivities_key"]
        else:
            self._conns_key = "connectivities"

        if "distances_key" in self._neighbors_dict:
            self._dists_key = self._neighbors_dict["distances_key"]
        else:
            self._dists_key = "distances"

        # Get actual matrices
        self._connectivities = None
        self._distances = None

        if self._conns_key in adata.obsp:
            self._connectivities = adata.obsp[self._conns_key]
        if self._dists_key in adata.obsp:
            self._distances = adata.obsp[self._dists_key]

    def __getitem__(self, key: str):
        if key == "distances":
            if "distances" not in self:
                msg = f"No {self._dists_key!r} in .obsp"
                raise KeyError(msg)
            return self._distances
        elif key == "connectivities":
            if "connectivities" not in self:
                msg = f"No {self._conns_key!r} in .obsp"
                raise KeyError(msg)
            return self._connectivities
        elif key == "connectivities_key":
            return self._conns_key
        elif key == "distances_key":
            return self._dists_key
        else:
            return self._neighbors_dict[key]

    def __contains__(self, key: str) -> bool:
        if key == "distances":
            return self._distances is not None
        elif key == "connectivities":
            return self._connectivities is not None
        else:
            return key in self._neighbors_dict


def _choose_representation(adata, use_rep=None, n_pcs=None):
    """Choose data representation for neighbor calculation."""
    if use_rep is None and n_pcs == 0:
        use_rep = "X"
    if use_rep is None:
        if adata.n_vars > 50:
            if "X_pca" in adata.obsm_keys():
                use_rep = "X_pca"
            else:
                # Compute PCA if not present
                from ._preprocess import pca
                logg.info("Computing PCA with default parameters...")
                pca(adata, n_comps=min(50, adata.shape[1] - 1))
                use_rep = "X_pca"
        else:
            use_rep = "X"

    if use_rep == "X":
        return adata.X
    elif use_rep in adata.obsm_keys():
        X = adata.obsm[use_rep]
        if n_pcs is not None:
            X = X[:, :n_pcs]
        return X
    else:
        msg = f"Did not find {use_rep} in `.obsm.keys()`. "
        raise ValueError(msg)


def get_igraph_from_adjacency(adjacency: CSBase, *, directed: bool = False):
    """Get igraph graph from adjacency matrix."""
    try:
        import igraph as ig
    except ImportError:
        msg = (
            "Please install the igraph package: "
            "`conda install -c conda-forge python-igraph` or "
            "`pip install igraph`."
        )
        raise ImportError(msg)

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets].A1 if len(sources) else []
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])
    g.add_edges(list(zip(sources, targets, strict=True)))
    try:
        g.es["weight"] = weights
    except KeyError:
        pass
    if g.vcount() != adjacency.shape[0]:
        logg.warning(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes."
        )
    return g


# Settings mock
class _Settings:
    """Mock settings object."""
    N_PCS = 50
    # Keep scanpy-compatible default for approximate kNN reproducibility.
    n_jobs = 1

settings = _Settings()

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping
    from typing import Any, Literal, NotRequired

    from anndata import AnnData
    from igraph import Graph

    # Type aliases
    _LegacyRandom = int | np.random.Generator | np.random.RandomState
    KnnTransformerLike = Any  # Any transformer with fit_transform method
    _MetricFn = Callable[[np.ndarray, np.ndarray], float]


RPForestDict = Mapping[str, Mapping[str, np.ndarray]]

N_DCS = 15  # default number of diffusion components
# Backwards compat, constants should be defined in only one place.
N_PCS = settings.N_PCS


class KwdsForTransformer(TypedDict):
    """Keyword arguments passed to a _KnownTransformer.

    IMPORTANT: when changing the parameters set here,
    update the “*ignored*” part in the parameter docs!
    """

    n_neighbors: int
    metric: _Metric | _MetricFn
    metric_params: Mapping[str, Any]
    random_state: _LegacyRandom
    n_jobs: int


class NeighborsParams(TypedDict):  # noqa: D101
    n_neighbors: int
    method: _Method
    random_state: _LegacyRandom
    metric: _Metric | _MetricFn
    metric_kwds: NotRequired[Mapping[str, Any]]
    use_rep: NotRequired[str]
    n_pcs: NotRequired[int]
    n_jobs: NotRequired[int]


@_doc_params(n_pcs=doc_n_pcs, use_rep=doc_use_rep)
def neighbors(  # noqa: PLR0913
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    *,
    use_rep: str | None = None,
    knn: bool = True,
    method: _Method = "torch",
    transformer: KnnTransformerLike | _KnownTransformer | None = None,
    metric: _Metric | _MetricFn = "euclidean",
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    random_state: _LegacyRandom = 0,
    n_jobs: int | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    r"""Compute nearest neighbors distance matrix and neighborhood graph with GPU acceleration.

    Compute the k-nearest neighbors graph and distance matrix for single-cell data.
    This implementation includes GPU-accelerated options using PyTorch for improved
    performance on large datasets.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_neighbors
        The size of local neighborhood (in terms of number of neighboring data
        points) used for manifold approximation. Larger values result in more
        global views of the manifold, while smaller values result in more local
        data being preserved. In general values should be in the range 2 to 100.
        If `knn` is `True`, number of nearest neighbors to be searched. If `knn`
        is `False`, a Gaussian kernel width is set to the distance of the
        `n_neighbors` neighbor.

        *ignored if ``transformer`` is an instance.*
    {n_pcs}
    {use_rep}
    knn
        If `True`, use a hard threshold to restrict the number of neighbors to
        `n_neighbors`, that is, consider a knn graph. Otherwise, use a Gaussian
        Kernel to assign low weights to neighbors more distant than the
        `n_neighbors` nearest neighbor.
    method
        Use 'umap' :cite:p:`McInnes2018` or 'gauss' (Gauss kernel following :cite:t:`Coifman2005`
        with adaptive width :cite:t:`Haghverdi2016`) for computing connectivities.
    transformer
        Approximate kNN search implementation following the API of
        :class:`~sklearn.neighbors.KNeighborsTransformer`.
        See :doc:`/how-to/knn-transformers` for more details.
        Also accepts the following known options:

        `None` (the default)
            Behavior depends on data size and hardware.
            For small data, we will calculate exact kNN, otherwise we use
            :class:`~pynndescent.pynndescent_.PyNNDescentTransformer`.
            If GPU is available and data size > 5000 cells, automatically use PyG KNN.
        `'pyg'`
            PyTorch Geometric KNN transformer (recommended for GPU).
            GPU-accelerated if CUDA is available, falls back to optimized CPU implementation.
            Typically 20-100× faster than other methods.
            Requires: `torch`, `torch-geometric` (optional: `torch-cluster` for GPU optimization).
        `'pynndescent'`
            :class:`~pynndescent.pynndescent_.PyNNDescentTransformer`
        `'rapids'`
            A transformer based on :class:`cuml.neighbors.NearestNeighbors`.

            .. deprecated:: 1.10.0
               Use :func:`rapids_singlecell.pp.neighbors` instead.
    metric
        A known metric’s name or a callable that returns a distance.

        *ignored if ``transformer`` is an instance.*
    metric_kwds
        Options for the metric.

        *ignored if ``transformer`` is an instance.*
    random_state
        A numpy random seed.

        *ignored if ``transformer`` is an instance.*
    n_jobs
        Number of parallel jobs used by neighbor search backends.
        If `None`, uses the internal default (`1`, scanpy-compatible).
    key_added
        If not specified, the neighbors data is stored in `.uns['neighbors']`,
        distances and connectivities are stored in `.obsp['distances']` and
        `.obsp['connectivities']` respectively.
        If specified, the neighbors data is added to .uns[key_added],
        distances are stored in `.obsp[key_added+'_distances']` and
        connectivities in `.obsp[key_added+'_connectivities']`.
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    Returns `None` if `copy=False`, else returns an `AnnData` object. Sets the following fields:

    `adata.obsp['distances' | key_added+'_distances']` : :class:`scipy.sparse.csr_matrix` (dtype `float`)
        Distance matrix of the nearest neighbors search. Each row (cell) has `n_neighbors`-1 non-zero entries. These are the distances to their `n_neighbors`-1 nearest neighbors (excluding the cell itself).
    `adata.obsp['connectivities' | key_added+'_connectivities']` : :class:`scipy.sparse._csr.csr_matrix` (dtype `float`)
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    `adata.uns['neighbors' | key_added]` : :class:`dict`
        neighbors parameters.

    Examples
    --------
    >>> import omicverse as ov
    >>> adata = ov.utils.read_adata('pbmc.h5ad')
    >>> # Basic usage
    >>> ov.pp.neighbors(adata, 20, metric="cosine")
    >>> # Provide your own transformer for more control and flexibility
    >>> from sklearn.neighbors import KNeighborsTransformer
    >>> transformer = KNeighborsTransformer(
    ...     n_neighbors=10, metric="manhattan", algorithm="kd_tree"
    ... )
    >>> ov.pp.neighbors(adata, transformer=transformer)
    >>> # now you can e.g. access the index: `transformer._tree`

    See Also
    --------
    :doc:`/how-to/knn-transformers`

    """
    print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} K-Nearest Neighbors Graph Construction:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Mode: {Colors.BOLD}{ov_settings.mode}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Neighbors: {Colors.BOLD}{n_neighbors}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Method: {Colors.BOLD}{method}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Metric: {Colors.BOLD}{metric}{Colors.ENDC}")
    if transformer is not None:
        transformer_name = transformer if isinstance(transformer, str) else type(transformer).__name__
        print(f"   {Colors.CYAN}Transformer: {Colors.BOLD}{transformer_name}{Colors.ENDC}")
    if use_rep is not None:
        print(f"   {Colors.CYAN}Representation: {Colors.BOLD}{use_rep}{Colors.ENDC}")
    if n_pcs is not None:
        print(f"   {Colors.CYAN}PCs used: {Colors.BOLD}{n_pcs}{Colors.ENDC}")
    
    start = logg.info("computing neighbors")
    adata = adata.copy() if copy else adata
    if adata.is_view:  # we shouldn't need this here...
        adata._init_as_actual(adata.copy())
    # Defend against Python id() reuse: if an old AnnData at this address
    # was cached, drop it before we cache a new graph for the current one.
    from ._gpu_cache import cache_invalidate
    cache_invalidate(adata)
    neighbors = Neighbors(adata)
    neighbors.compute_neighbors(
        n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        knn=knn,
        method=method,
        transformer=transformer,
        metric=metric,
        metric_kwds=metric_kwds,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    if key_added is None:
        key_added = "neighbors"
        conns_key = "connectivities"
        dists_key = "distances"
    else:
        conns_key = key_added + "_connectivities"
        dists_key = key_added + "_distances"

    adata.uns[key_added] = {}

    neighbors_dict = adata.uns[key_added]

    neighbors_dict["connectivities_key"] = conns_key
    neighbors_dict["distances_key"] = dists_key

    neighbors_dict["params"] = NeighborsParams(
        n_neighbors=neighbors.n_neighbors,
        method=method,
        random_state=random_state,
        metric=metric,
    )
    if n_jobs is not None:
        neighbors_dict["params"]["n_jobs"] = n_jobs
    if metric_kwds:
        neighbors_dict["params"]["metric_kwds"] = metric_kwds
    if use_rep is not None:
        neighbors_dict["params"]["use_rep"] = use_rep
    if n_pcs is not None:
        neighbors_dict["params"]["n_pcs"] = n_pcs

    # 关键：先规范化再写入 obsp
    D = _to_sorted_csr(neighbors.distances)
    C = _to_sorted_csr(neighbors.connectivities)

    # （可选）确保形状一致
    n = adata.n_obs
    if D is not None and D.shape != (n, n):
        raise ValueError(f"distances shape {D.shape} != ({n},{n})")
    if C is not None and C.shape != (n, n):
        raise ValueError(f"connectivities shape {C.shape} != ({n},{n})")

    adata.obsp[dists_key] = D
    adata.obsp[conns_key] = C

    # Register device-resident COO tensors in the cache (keyed against the
    # final adata.obsp matrix so downstream consumers compare the right
    # source). Skipped for non-GPU paths.
    gpu_coo = getattr(neighbors, "_gpu_coo", None)
    if gpu_coo is not None:
        from ._gpu_cache import cache_set
        cache_set(adata, "connectivities", gpu_coo, source_obj=adata.obsp[conns_key])

    # adata.obsp[dists_key] = neighbors.distances
    # adata.obsp[conns_key] = neighbors.connectivities

    if neighbors.rp_forest is not None:
        neighbors_dict["rp_forest"] = neighbors.rp_forest
    print(f"\n{Colors.GREEN}{EMOJI['done']} KNN Graph Construction Completed Successfully!{Colors.ENDC}")
    n_cells = adata.shape[0]
    print(f"   {Colors.GREEN}✓ Processed: {Colors.BOLD}{n_cells:,}{Colors.ENDC}{Colors.GREEN} cells with {Colors.BOLD}{neighbors.n_neighbors}{Colors.ENDC}{Colors.GREEN} neighbors each{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Results added to AnnData object:{Colors.ENDC}")
    print(f"     {Colors.CYAN}• '{key_added}': {Colors.BOLD}Neighbors metadata{Colors.ENDC}{Colors.CYAN} (adata.uns){Colors.ENDC}")
    print(f"     {Colors.CYAN}• '{dists_key}': {Colors.BOLD}Distance matrix{Colors.ENDC}{Colors.CYAN} (adata.obsp){Colors.ENDC}")
    print(f"     {Colors.CYAN}• '{conns_key}': {Colors.BOLD}Connectivity matrix{Colors.ENDC}{Colors.CYAN} (adata.obsp){Colors.ENDC}")
    
    
    adata.uns[key_added]=neighbors_dict
    return adata if copy else None


class FlatTree(NamedTuple):  # noqa: D101
    hyperplanes: None
    offsets: None
    children: None
    indices: None


def _backwards_compat_get_full_X_diffmap(adata: AnnData) -> np.ndarray:
    if "X_diffmap0" in adata.obs:
        return np.c_[adata.obs["X_diffmap0"].values[:, None], adata.obsm["X_diffmap"]]
    else:
        return adata.obsm["X_diffmap"]


def _backwards_compat_get_full_eval(adata: AnnData):
    if "X_diffmap0" in adata.obs:
        return np.r_[1, adata.uns["diffmap_evals"]]
    else:
        return adata.uns["diffmap_evals"]


def _make_forest_dict(forest):
    d = {}
    props = ("hyperplanes", "offsets", "children", "indices")
    for prop in props:
        d[prop] = {}
        sizes = np.fromiter(
            (getattr(tree, prop).shape[0] for tree in forest), dtype=int
        )
        d[prop]["start"] = np.zeros_like(sizes)
        if prop == "offsets":
            dims = sizes.sum()
        else:
            dims = (sizes.sum(), getattr(forest[0], prop).shape[1])
        dtype = getattr(forest[0], prop).dtype
        dat = np.empty(dims, dtype=dtype)
        start = 0
        for i, size in enumerate(sizes):
            d[prop]["start"][i] = start
            end = start + size
            dat[start:end] = getattr(forest[i], prop)
            start = end
        d[prop]["data"] = dat
    return d

def _to_sorted_csr(X):
    import scipy.sparse as sp
    if X is None:
        return None
    if not sp.issparse(X):
        raise TypeError("neighbors.* must be a scipy.sparse matrix")
    X = X.tocsr(copy=True)        # 转 CSR
    X.sum_duplicates()            # 合并重复 (i,j)
    X.sort_indices()              # 行内列索引递增
    # 可选：清理 NaN/Inf，避免写盘后再炸
    if np.isnan(X.data).any() or np.isinf(X.data).any():
        X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0)
        X.eliminate_zeros()
    return X

class OnFlySymMatrix:
    """Emulate a matrix where elements are calculated on the fly."""

    def __init__(
        self,
        get_row: Callable[[Any], np.ndarray],
        shape: tuple[int, int],
        *,
        DC_start: int = 0,
        DC_end: int = -1,
        rows: MutableMapping[Any, np.ndarray] | None = None,
        restrict_array: np.ndarray | None = None,
    ):
        self.get_row = get_row
        self.shape = shape
        self.DC_start = DC_start
        self.DC_end = DC_end
        self.rows = {} if rows is None else rows
        self.restrict_array = restrict_array  # restrict the array to a subset

    def __getitem__(self, index):  # noqa: D105
        if isinstance(index, int | np.integer):
            if self.restrict_array is None:
                glob_index = index
            else:
                # map the index back to the global index
                glob_index = self.restrict_array[index]
            if glob_index not in self.rows:
                self.rows[glob_index] = self.get_row(glob_index)
            row = self.rows[glob_index]
            if self.restrict_array is None:
                return row
            else:
                return row[self.restrict_array]
        else:
            if self.restrict_array is None:
                glob_index_0, glob_index_1 = index
            else:
                glob_index_0 = self.restrict_array[index[0]]
                glob_index_1 = self.restrict_array[index[1]]
            if glob_index_0 not in self.rows:
                self.rows[glob_index_0] = self.get_row(glob_index_0)
            return self.rows[glob_index_0][glob_index_1]

    def restrict(self, index_array):
        """Generate a view restricted to a subset of indices."""
        new_shape = index_array.shape[0], index_array.shape[0]
        return OnFlySymMatrix(
            self.get_row,
            new_shape,
            DC_start=self.DC_start,
            DC_end=self.DC_end,
            rows=self.rows,
            restrict_array=index_array,
        )


class Neighbors:
    """Data represented as graph of nearest neighbors.

    Represent a data matrix as a graph of nearest neighbor relations (edges)
    among data points (nodes).

    Parameters
    ----------
    adata
        Annotated data object.
    n_dcs
        Number of diffusion components to use.
    neighbors_key
        Where to look in `.uns` and `.obsp` for neighbors data

    """

    @old_positionals("n_dcs", "neighbors_key")
    def __init__(  # noqa: PLR0912, PLR0915
        self,
        adata: AnnData,
        *,
        n_dcs: int | None = None,
        neighbors_key: str | None = None,
    ):
        self._adata = adata
        self._init_iroot()
        # use the graph in adata
        info_str = ""
        self.knn: bool | None = None
        self._distances: np.ndarray | CSRBase | None = None
        self._connectivities: np.ndarray | CSRBase | None = None
        self._transitions_sym: np.ndarray | CSRBase | None = None
        self._number_connected_components: int | None = None
        self._rp_forest: RPForestDict | None = None
        if neighbors_key is None:
            neighbors_key = "neighbors"
        if neighbors_key in adata.uns:
            neighbors = NeighborsView(adata, neighbors_key)
            if "distances" in neighbors:
                self.knn = isinstance(neighbors["distances"], CSBase)
                self._distances = neighbors["distances"]
            if "connectivities" in neighbors:
                self.knn = isinstance(neighbors["connectivities"], CSBase)
                self._connectivities = neighbors["connectivities"]
            if "rp_forest" in neighbors:
                self._rp_forest = neighbors["rp_forest"]
            if "params" in neighbors:
                self.n_neighbors = neighbors["params"]["n_neighbors"]
            else:

                def count_nonzero(a: np.ndarray | CSRBase) -> int:
                    return (
                        a.count_nonzero()
                        if isinstance(a, CSRBase)
                        else np.count_nonzero(a)
                    )

                # estimating n_neighbors
                if self._connectivities is None:
                    self.n_neighbors = int(
                        count_nonzero(self._distances) / self._distances.shape[0]
                    )
                else:
                    self.n_neighbors = int(
                        count_nonzero(self._connectivities)
                        / self._connectivities.shape[0]
                        / 2
                    )
            info_str += "`.distances` `.connectivities` "
            self._number_connected_components = 1
            if isinstance(self._connectivities, CSBase):
                from scipy.sparse.csgraph import connected_components

                self._connected_components = connected_components(self._connectivities)
                self._number_connected_components = self._connected_components[0]
        if "X_diffmap" in adata.obsm_keys():
            self._eigen_values = _backwards_compat_get_full_eval(adata)
            self._eigen_basis = _backwards_compat_get_full_X_diffmap(adata)
            if n_dcs is not None:
                if n_dcs > len(self._eigen_values):
                    msg = (
                        f"Cannot instantiate using `n_dcs`={n_dcs}. "
                        "Compute diffmap/spectrum with more components first."
                    )
                    raise ValueError(msg)
                self._eigen_values = self._eigen_values[:n_dcs]
                self._eigen_basis = self._eigen_basis[:, :n_dcs]
            self.n_dcs = len(self._eigen_values)
            info_str += "`.eigen_values` `.eigen_basis` `.distances_dpt`"
        else:
            self._eigen_values = None
            self._eigen_basis = None
            self.n_dcs = None
        if info_str != "":
            logg.debug(f"    initialized {info_str}")

    @property
    def rp_forest(self) -> RPForestDict | None:
        """PyNNDescent index."""
        return self._rp_forest

    @property
    def distances(self) -> np.ndarray | CSRBase | None:
        """Distances between data points (sparse matrix)."""
        return self._distances

    @property
    def connectivities(self) -> np.ndarray | CSRBase | None:
        """Connectivities between data points (sparse matrix)."""
        return self._connectivities

    @property
    def transitions(self) -> np.ndarray | CSRBase:
        """Transition matrix (sparse matrix).

        Is conjugate to the symmetrized transition matrix via::

            self.transitions = self.Z * self.transitions_sym / self.Z

        where ``self.Z`` is the diagonal matrix storing the normalization of the
        underlying kernel matrix.

        Notes
        -----
        This has not been tested, in contrast to `transitions_sym`.

        """
        Zinv = (
            self.Z.power(-1)
            if isinstance(self.Z, SpBase)  # can be DIA matrix
            else np.diag(1.0 / np.diag(self.Z))
        )
        return self.Z @ self.transitions_sym @ Zinv

    @property
    def transitions_sym(self) -> np.ndarray | CSRBase | None:
        """Symmetrized transition matrix (sparse matrix).

        Is conjugate to the transition matrix via::

            self.transitions_sym = self.Z / self.transitions * self.Z

        where ``self.Z`` is the diagonal matrix storing the normalization of the
        underlying kernel matrix.
        """
        return self._transitions_sym

    @property
    def eigen_values(self) -> np.ndarray:
        """Eigen values of transition matrix."""
        return self._eigen_values

    @property
    def eigen_basis(self) -> np.ndarray:
        """Eigen basis of transition matrix."""
        return self._eigen_basis

    @property
    def distances_dpt(self) -> OnFlySymMatrix:
        """DPT distances.

        This is yields :cite:p:`Haghverdi2016`, Eq. 15 from the supplement with the
        extensions of :cite:p:`Wolf2019`, supplement on random-walk based distance
        measures.
        """
        return OnFlySymMatrix(self._get_dpt_row, shape=self._adata.shape)

    def to_igraph(self) -> Graph:
        """Generate igraph from connectiviies."""
        return get_igraph_from_adjacency(self.connectivities)

    @_doc_params(n_pcs=doc_n_pcs, use_rep=doc_use_rep)
    def compute_neighbors(  # noqa: PLR0912
        self,
        n_neighbors: int = 30,
        n_pcs: int | None = None,
        *,
        use_rep: str | None = None,
        knn: bool = True,
        method: _Method | None = "torch",
        transformer: KnnTransformerLike | _KnownTransformer | None = None,
        metric: _Metric | _MetricFn = "euclidean",
        metric_kwds: Mapping[str, Any] = MappingProxyType({}),
        random_state: _LegacyRandom = 0,
        n_jobs: int | None = None,
    ) -> None:
        """Compute distances and connectivities of neighbors.

        Parameters
        ----------
        n_neighbors
            Use this number of nearest neighbors.
        {n_pcs}
        {use_rep}
        knn
            Restrict result to `n_neighbors` nearest neighbors.
        method
            See :func:`scanpy.pp.neighbors`.
            If `None`, skip calculating connectivities.

        Returns
        -------
        Writes sparse graph attributes `.distances` and,
        if `method` is not `None`, `.connectivities`.

        """
        print(f"   {Colors.GREEN}{EMOJI['start']} Computing neighbor distances...{Colors.ENDC}")
        start_neighbors = logg.debug("computing neighbors")
        if transformer is not None and not isinstance(transformer, str):
            n_neighbors = transformer.get_params()["n_neighbors"]
        elif n_neighbors > self._adata.shape[0]:  # very small datasets
            n_neighbors = 1 + int(0.5 * self._adata.shape[0])
            print(f"   {EMOJI['warning']} {Colors.WARNING}Dataset too small: adjusting to {Colors.BOLD}{n_neighbors}{Colors.ENDC}{Colors.WARNING} neighbors{Colors.ENDC}")
            logg.warning(f"n_obs too small: adjusting to `n_neighbors = {n_neighbors}`")

        # Match scanpy by default, but allow explicit override.
        effective_n_jobs = settings.n_jobs if n_jobs is None else n_jobs

        # default keyword arguments when `transformer` is not an instance
        transformer_kwds_default = KwdsForTransformer(
            n_neighbors=n_neighbors,
            metric=metric,
            metric_params=metric_kwds,  # most use _params, not _kwds
            random_state=random_state,
            n_jobs=effective_n_jobs,
        )
        method, transformer, shortcut = self._handle_transformer(
            method, transformer, knn=knn, kwds=transformer_kwds_default
        )

        if self._adata.shape[0] >= 10000 and not knn:
            print(f"   {EMOJI['warning']} {Colors.WARNING}Large dataset without knn=True may require significant memory{Colors.ENDC}")
            logg.warning("Using high n_obs without `knn=True` takes a lot of memory...")
        # do not use the cached rp_forest
        self._rp_forest = None
        self.n_neighbors = n_neighbors
        self.knn = knn
        X = _choose_representation(self._adata, use_rep=use_rep, n_pcs=n_pcs)
        self._distances = transformer.fit_transform(X)
        knn_indices, knn_distances = _get_indices_distances_from_sparse_matrix(
            self._distances, n_neighbors
        )
        if shortcut:
            # self._distances is a sparse matrix with a diag of 1, fix that
            self._distances[np.diag_indices_from(self.distances)] = 0
            if knn:  # remove too far away entries in self._distances
                self._distances = _get_sparse_matrix_from_indices_distances(
                    knn_indices, knn_distances, keep_self=False
                )
            else:  # convert to dense
                self._distances = self._distances.toarray()
        if index := getattr(transformer, "index_", None):
            from pynndescent import NNDescent

            if isinstance(index, NNDescent):
                # very cautious here
                # TODO catch the correct exception
                with contextlib.suppress(Exception):
                    self._rp_forest = _make_forest_dict(index)
        start_connect = logg.debug("computed neighbors", time=start_neighbors)
        print(f"   {Colors.GREEN}{EMOJI['start']} Computing connectivity matrix...{Colors.ENDC}")
        if method == "torch":
            print(f"   {Colors.CYAN}💡 Using GPU-optimized connectivity computation{Colors.ENDC}")
        elif method == "umap":
            print(f"   {Colors.CYAN}💡 Using UMAP-style connectivity{Colors.ENDC}")
        elif method == "gauss":
            print(f"   {Colors.CYAN}💡 Using Gaussian kernel connectivity{Colors.ENDC}")

        if method == "umap":
            from ._connectivity import umap
            self._connectivities = umap(
                knn_indices,
                knn_distances,
                n_obs=self._adata.shape[0],
                n_neighbors=self.n_neighbors,
            )
        elif method == "gauss":
            from ._connectivity import gauss
            self._connectivities = gauss(
                self._distances, self.n_neighbors, knn=self.knn
            )
        elif method == "torch":
            import torch
            from ..external.umap_pytorch.fuzzy_gpu import fuzzy_simplicial_set_gpu
            from scipy.sparse import coo_matrix

            device = normalize_torch_device()
            print(f"   {Colors.CYAN}Using device: {Colors.BOLD}{device}{Colors.ENDC}")

            # Build the fuzzy simplicial set on device once. Convert to scipy
            # CSR for adata.obsp (existing contract) and stash the GPU tensors
            # on self so the outer neighbors() can register them in the cache
            # for downstream ov.pp.umap / ov.pp.leiden to reuse.
            knn_i = torch.as_tensor(knn_indices, dtype=torch.long, device=device)
            knn_d = torch.as_tensor(knn_distances, dtype=torch.float32, device=device)
            rows, cols, vals = fuzzy_simplicial_set_gpu(
                knn_i, knn_d, n_neighbors=int(self.n_neighbors),
            )
            n_obs = self._adata.shape[0]
            self._connectivities = coo_matrix(
                (vals.detach().cpu().numpy(),
                 (rows.detach().cpu().numpy(), cols.detach().cpu().numpy())),
                shape=(n_obs, n_obs),
            ).tocsr()
            self._gpu_coo = (rows, cols, vals, n_obs)
        elif method is not None:
            msg = f"{method!r} should have been coerced in _handle_transform_args"
            raise AssertionError(msg)
        self._number_connected_components = 1
        if isinstance(self._connectivities, CSBase):
            from scipy.sparse.csgraph import connected_components

            self._connected_components = connected_components(self._connectivities)
            self._number_connected_components = self._connected_components[0]
        if method is not None:
            if self._number_connected_components > 1:
                print(f"   {EMOJI['warning']} {Colors.WARNING}Found {Colors.BOLD}{self._number_connected_components}{Colors.ENDC}{Colors.WARNING} disconnected components in the graph{Colors.ENDC}")
            else:
                print(f"   {Colors.GREEN}✓ Graph is fully connected{Colors.ENDC}")
            logg.debug("computed connectivities", time=start_connect)

    def _handle_transformer(
        self,
        method: _Method | Literal["gauss", "torch"] | None,
        transformer: KnnTransformerLike | _KnownTransformer | None,
        *,
        knn: bool,
        kwds: KwdsForTransformer,
    ) -> tuple[_Method | None, KnnTransformerLike, bool]:
        """Return effective `method` and transformer.

        `method` will be coerced to `'gauss'` or `'umap'`.
        `transformer` is coerced from a str or instance to an instance class.

        If `transformer` is `None` and there are few data points,
        `transformer` will be set to a brute force
        :class:`~sklearn.neighbors.KNeighborsTransformer`.

        If `transformer` is `None` and there are many data points,
        `transformer` will be set like `umap` does (i.e. to a
        ~`pynndescent.PyNNDescentTransformer` with custom `n_trees` and `n_iter`).
        """
        # If the caller asked for `method='torch'` and didn't pin the
        # transformer themselves, default to the GPU chunked-matmul KNN
        # (TorchKNNTransformer) so 'torch' actually means torch end-to-end.
        # Other metrics, no-CUDA, or huge memory pressure still fall through
        # to PyNNDescent below.
        if (
            method == "torch"
            and transformer is None
            and kwds["metric"] == "euclidean"
            and TORCH_AVAILABLE
            and torch.cuda.is_available()
        ):
            transformer = "pyg"

        # legacy logic
        use_dense_distances = (
            kwds["metric"] == "euclidean" and self._adata.n_obs < 8192
        ) or not knn
        shortcut = transformer == "sklearn" or (
            transformer is None and (use_dense_distances or self._adata.n_obs < 4096)
        )

        # Coerce `method` to 'gauss' or 'umap'
        if method == "rapids":
            if transformer is not None:
                msg = "Can’t specify both `method = 'rapids'` and `transformer`."
                raise ValueError(msg)
            method = "umap"
            transformer = "rapids"
        elif (
            method not in (methods := get_literal_vals(_Method)) and method is not None
        ):
            msg = f"`method` needs to be one of {methods}."
            raise ValueError(msg)

        # Validate `knn`. ``torch`` keeps its own identity (not coerced to
        # ``umap``) so the GPU fuzzy_simplicial_set branch in
        # ``compute_neighbors`` fires instead of umap-learn's CPU one; this
        # is what the ``cpu-gpu-mixed`` mode was supposed to do all along.
        conn_method = method if method in {"gauss", None, "torch"} else "umap"
        if not knn and not (conn_method == "gauss" and transformer is None):
            # “knn=False” seems to be only intended for method “gauss”
            msg = f"`method = {method!r} only with `knn = True`."
            raise ValueError(msg)

        # Coerce `transformer` to an instance
        if shortcut:
            from sklearn.neighbors import KNeighborsTransformer

            assert transformer in {None, "sklearn"}
            n_neighbors = self._adata.n_obs - 1
            if knn:  # only obey n_neighbors arg if knn set
                n_neighbors = min(n_neighbors, kwds["n_neighbors"])
            transformer = KNeighborsTransformer(
                algorithm="brute",
                n_jobs=kwds["n_jobs"],
                n_neighbors=n_neighbors,
                metric=kwds["metric"],
                metric_params=dict(kwds["metric_params"]),  # needs dict
                # no random_state
            )
        elif transformer == "pyg":
            # Use PyTorch Geometric KNN
            if not TORCH_AVAILABLE:
                msg = (
                    "PyG transformer requires PyTorch. "
                    "Install with: pip install torch torch-geometric"
                )
                raise ImportError(msg)

            try:
                from .pyg_knn_implementation import TorchKNNTransformer
            except ImportError as e:
                msg = (
                    "PyG transformer implementation not found. "
                    "Make sure pyg_knn_implementation.py is in omicverse/pp/"
                )
                raise ImportError(msg) from e

            device = normalize_torch_device()
            print(f"   {Colors.CYAN}💡 Using PyTorch Geometric KNN on {Colors.BOLD}{device}{Colors.ENDC}")

            transformer = TorchKNNTransformer(
                n_neighbors=kwds["n_neighbors"],
                device=device
            )
            shortcut = False  # Don't use sklearn shortcut

        elif transformer is None or transformer == "pynndescent":
            

            # Use PyNNDescent if not using PyG
            if not isinstance(transformer, object) or transformer in [None, "pynndescent"]:
                from pynndescent import PyNNDescentTransformer

                kwds = kwds.copy()
                kwds["metric_kwds"] = kwds.pop("metric_params")
                if transformer is None:
                    # Use defaults from UMAP's `nearest_neighbors` function
                    kwds.update(
                        n_jobs=kwds["n_jobs"],
                        n_trees=min(64, 5 + round((self._adata.n_obs) ** 0.5 / 20.0)),
                        n_iters=max(5, round(np.log2(self._adata.n_obs))),
                    )
                transformer = PyNNDescentTransformer(**kwds)
        elif transformer == "rapids":
            msg = (
                "`transformer='rapids'` is deprecated. "
                "Use `rapids_singlecell.pp.neighbors` instead."
            )
            warn(msg, FutureWarning, stacklevel=3)
            try:
                from rapids_singlecell._neighbors import RapidsKNNTransformer
                transformer = RapidsKNNTransformer(**kwds)
            except ImportError:
                msg = (
                    "RAPIDS transformer requires `rapids_singlecell` package. "
                    "Install with: `pip install rapids-singlecell`"
                )
                raise ImportError(msg)
        elif isinstance(transformer, str):
            msg = (
                f"Unknown transformer: {transformer}. "
                f"Try passing a class or one of {get_literal_vals(_KnownTransformer)}"
            )
            raise ValueError(msg)
        # else `transformer` is probably an instance
        return conn_method, transformer, shortcut

    @old_positionals("density_normalize")
    def compute_transitions(self, *, density_normalize: bool = True):
        """Compute transition matrix.

        Parameters
        ----------
        density_normalize
            The density rescaling of Coifman and Lafon (2006): Then only the
            geometry of the data matters, not the sampled density.

        Returns
        -------
        Makes attributes `.transitions_sym` and `.transitions` available.

        """
        print(f"   {Colors.GREEN}{EMOJI['start']} Computing transition matrix...{Colors.ENDC}")
        if density_normalize:
            print(f"   {Colors.CYAN}Density normalization: {Colors.BOLD}enabled{Colors.ENDC}")
        start = logg.info("computing transitions")
        W = self._connectivities
        # density normalization as of Coifman et al. (2005)
        # ensures that kernel matrix is independent of sampling density
        if density_normalize:
            # q[i] is an estimate for the sampling density at point i
            # it's also the degree of the underlying graph
            q = np.asarray(W.sum(axis=0))
            if not isinstance(W, CSBase):
                Q = np.diag(1.0 / q)
            else:
                Q = sparse.spdiags(1.0 / q, 0, W.shape[0], W.shape[0])
            K = Q @ W @ Q
        else:
            K = W

        # z[i] is the square root of the row sum of K
        z = np.sqrt(np.asarray(K.sum(axis=0)))
        if not isinstance(K, CSBase):
            self.Z = np.diag(1.0 / z)
        else:
            self.Z = sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
        self._transitions_sym = self.Z @ K @ self.Z
        print(f"   {Colors.GREEN}✓ Transition matrix computed successfully{Colors.ENDC}")
        logg.info("    finished", time=start)

    def compute_eigen(
        self,
        n_comps: int = 15,
        sym: bool | None = None,
        sort: Literal["decrease", "increase"] = "decrease",
        random_state: _LegacyRandom = 0,
    ):
        """Compute eigen decomposition of transition matrix.

        Parameters
        ----------
        n_comps
            Number of eigenvalues/vectors to be computed, set `n_comps = 0` if
            you need all eigenvectors.
        sym
            Instead of computing the eigendecomposition of the assymetric
            transition matrix, computed the eigendecomposition of the symmetric
            Ktilde matrix.
        random_state
            A numpy random seed

        Returns
        -------
        Writes the following attributes.

        eigen_values : :class:`~numpy.ndarray`
            Eigenvalues of transition matrix.
        eigen_basis : :class:`~numpy.ndarray`
            Matrix of eigenvectors (stored in columns).  `.eigen_basis` is
            projection of data matrix on right eigenvectors, that is, the
            projection on the diffusion components.  these are simply the
            components of the right eigenvectors and can directly be used for
            plotting.

        """
        np.set_printoptions(precision=10)
        if self._transitions_sym is None:
            msg = "Run `.compute_transitions` first."
            raise ValueError(msg)
        matrix = self._transitions_sym
        # compute the spectrum
        if n_comps == 0:
            evals, evecs = scipy.linalg.eigh(matrix)
        else:
            n_comps = min(matrix.shape[0] - 1, n_comps)
            # ncv = max(2 * n_comps + 1, int(np.sqrt(matrix.shape[0])))
            ncv = None
            which = "LM" if sort == "decrease" else "SM"
            # it pays off to increase the stability with a bit more precision
            matrix = matrix.astype(np.float64)

            # Setting the random initial vector
            random_state = check_random_state(random_state)
            v0 = random_state.standard_normal(matrix.shape[0])
            evals, evecs = sparse.linalg.eigsh(
                matrix, k=n_comps, which=which, ncv=ncv, v0=v0
            )
            evals, evecs = evals.astype(np.float32), evecs.astype(np.float32)
        if sort == "decrease":
            evals = evals[::-1]
            evecs = evecs[:, ::-1]
        print(f"   {Colors.GREEN}✓ Computed {Colors.BOLD}{len(evals)}{Colors.ENDC}{Colors.GREEN} eigenvalues/eigenvectors{Colors.ENDC}")
        print(f"   {Colors.CYAN}Top eigenvalues: {Colors.BOLD}{evals[:min(5, len(evals))]}{Colors.ENDC}")
        logg.info(f"    eigenvalues of transition matrix\n{indent(str(evals), '    ')}")
        if self._number_connected_components > len(evals) / 2:
            print(f"   {EMOJI['warning']} {Colors.WARNING}Transition matrix has many disconnected components!{Colors.ENDC}")
            logg.warning("Transition matrix has many disconnected components!")
        self._eigen_values = evals
        self._eigen_basis = evecs

    def _init_iroot(self):
        self.iroot = None
        # set iroot directly
        if "iroot" in self._adata.uns:
            if self._adata.uns["iroot"] >= self._adata.n_obs:
                print(f"   {EMOJI['warning']} {Colors.WARNING}Root cell index {Colors.BOLD}{self._adata.uns['iroot']}{Colors.ENDC}{Colors.WARNING} does not exist for {Colors.BOLD}{self._adata.n_obs}{Colors.ENDC}{Colors.WARNING} samples, ignoring{Colors.ENDC}")
                msg = (
                    f"Root cell index {self._adata.uns['iroot']} does not "
                    f"exist for {self._adata.n_obs} samples. It's ignored."
                )
                logg.warning(msg)
            else:
                self.iroot = self._adata.uns["iroot"]
            return
        # set iroot via xroot
        xroot = None
        if "xroot" in self._adata.uns:
            xroot = self._adata.uns["xroot"]
        elif "xroot" in self._adata.var:
            xroot = self._adata.var["xroot"]
        # see whether we can set self.iroot using the full data matrix
        if xroot is not None and xroot.size == self._adata.shape[1]:
            self._set_iroot_via_xroot(xroot)

    def _get_dpt_row(self, i: int) -> np.ndarray:
        mask = None
        if self._number_connected_components > 1:
            label = self._connected_components[1][i]
            mask = self._connected_components[1] == label
        row = sum(
            (
                self.eigen_values[j]
                / (1 - self.eigen_values[j])
                * (self.eigen_basis[i, j] - self.eigen_basis[:, j])
            )
            ** 2
            # account for float32 precision
            for j in range(self.eigen_values.size)
            if self.eigen_values[j] < 0.9994
        )
        # thanks to Marius Lange for pointing Alex to this:
        # we will likely remove the contributions from the stationary state below when making
        # backwards compat breaking changes, they originate from an early implementation in 2015
        # they never seem to have deteriorated results, but also other distance measures (see e.g.
        # PAGA paper) don't have it, which makes sense
        row += sum(
            (self.eigen_basis[i, k] - self.eigen_basis[:, k]) ** 2
            for k in range(self.eigen_values.size)
            if self.eigen_values[k] >= 0.9994
        )
        if mask is not None:
            row[~mask] = np.inf
        return np.sqrt(row)

    def _set_pseudotime(self):
        """Return pseudotime with respect to root point."""
        self.pseudotime = self.distances_dpt[self.iroot].copy()
        self.pseudotime /= np.max(self.pseudotime[self.pseudotime < np.inf])

    def _set_iroot_via_xroot(self, xroot: np.ndarray):
        """Determine the index of the root cell.

        Given an expression vector, find the observation index that is closest
        to this vector.

        Parameters
        ----------
        xroot
            Vector that marks the root cell, the vector storing the initial
            condition, only relevant for computing pseudotime.

        """
        if self._adata.shape[1] != xroot.size:
            msg = "The root vector you provided does not have the correct dimension."
            raise ValueError(msg)
        # this is the squared distance
        dsqroot = 1e10
        iroot = 0
        for i in range(self._adata.shape[0]):
            diff = self._adata.X[i, :] - xroot
            dsq = diff @ diff
            if dsq < dsqroot:
                dsqroot = dsq
                iroot = i
                if np.sqrt(dsqroot) < 1e-10:
                    break
        logg.debug(f"setting root index to {iroot}")
        if self.iroot is not None and iroot != self.iroot:
            print(f"   {EMOJI['warning']} {Colors.WARNING}Changing root index from {Colors.BOLD}{self.iroot}{Colors.ENDC}{Colors.WARNING} to {Colors.BOLD}{iroot}{Colors.ENDC}")
            logg.warning(f"Changing index of iroot from {self.iroot} to {iroot}.")
        self.iroot = iroot
