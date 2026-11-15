from __future__ import annotations

import warnings
from contextlib import contextmanager, suppress
from datetime import datetime
from typing import TYPE_CHECKING
import importlib.util

import numpy as np
import pandas as pd
from natsort import natsorted


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence
    from typing import Literal

    from anndata import AnnData
    from numpy.typing import NDArray

    from ._compat import CSBase

    try:  # sphinx-autodoc-typehints + optional dependency
        from leidenalg.VertexPartition import MutableVertexPartition
    except ImportError:
        if not TYPE_CHECKING:
            MutableVertexPartition = type("MutableVertexPartition", (), {})
            MutableVertexPartition.__module__ = "leidenalg.VertexPartition"


# ============================================================================
# Utility Functions (ported from scanpy to remove dependency)
# ============================================================================

def ensure_igraph() -> None:
    """Check if igraph is installed."""
    if importlib.util.find_spec("igraph"):
        return
    msg = (
        "Please install the igraph package: "
        "`conda install -c conda-forge python-igraph` or "
        "`pip install igraph`."
    )
    raise ImportError(msg)


def dematrix(x):
    """Convert numpy matrix to array."""
    if isinstance(x, np.matrix):
        return x.A
    return x


def _choose_graph(
    adata: AnnData, obsp: str | None, neighbors_key: str | None
) -> "CSBase":
    """Choose connectivities from neighbors or another obsp entry."""
    if obsp is not None and neighbors_key is not None:
        msg = "You can't specify both obsp, neighbors_key. Please select only one."
        raise ValueError(msg)

    if obsp is not None:
        return adata.obsp[obsp]
    else:
        # Simple implementation without NeighborsView
        if neighbors_key is None:
            connectivities_key = "connectivities"
        else:
            if neighbors_key not in adata.uns:
                msg = f"neighbors_key {neighbors_key!r} not found in adata.uns"
                raise ValueError(msg)
            neighbors_dict = adata.uns[neighbors_key]
            if "connectivities_key" in neighbors_dict:
                connectivities_key = neighbors_dict["connectivities_key"]
            else:
                connectivities_key = "connectivities"

        if connectivities_key not in adata.obsp:
            msg = (
                "You need to run `pp.neighbors` first to compute a neighborhood graph."
            )
            raise ValueError(msg)
        return adata.obsp[connectivities_key]


class _RNGIgraph:
    """Random number generator for igraph avoiding global seed modification."""

    def __init__(self, random_state: int | np.random.RandomState = 0) -> None:
        if isinstance(random_state, np.random.RandomState):
            self._rng = random_state
        else:
            self._rng = np.random.RandomState(random_state)

    def getrandbits(self, k: int) -> int:
        # Same draw as scanpy's _RNGIgraph, so flavor='igraph' labels match scanpy
        # for the same seed (randint(0, 2**k) consumed the stream differently).
        return self._rng.tomaxint() & ((1 << k) - 1)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b + 1)

    def __getattr__(self, attr: str):
        # For igraph compatibility, map 'gauss' to 'normal'
        if attr == "gauss":
            return self._rng.normal
        return getattr(self._rng, attr)


@contextmanager
def set_igraph_random_state(
    random_state: int | np.random.RandomState,
) -> Generator[None, None, None]:
    """Context manager for igraph random number generator."""
    ensure_igraph()
    import igraph
    import random as stdlib_random

    rng = _RNGIgraph(random_state)
    try:
        igraph.set_random_number_generator(rng)
        yield None
    finally:
        igraph.set_random_number_generator(stdlib_random)


def warn_once(msg: str, category: type[Warning] = UserWarning, stacklevel: int = 2):
    """Warn only once for a given message."""
    # Simple implementation: just issue warning every time
    # To truly warn once, you'd need to track warnings issued
    warnings.warn(msg, category=category, stacklevel=stacklevel)


# Simple logging functions to replace scanpy.logging
_log_start_time = None

def info(msg: str, *, time: datetime | None = None, deep: str | None = None) -> datetime:
    """Log info message."""
    now = datetime.now()
    if time is not None:
        time_passed = now - time
        print(f"{msg} ({time_passed.total_seconds():.2f}s)")
    else:
        print(msg)
    if deep:
        print(f"    {deep}")
    return now


# ============================================================================
# Helper Functions
# ============================================================================

def rename_groups(
    adata: AnnData,
    restrict_key: str,
    *,
    key_added: str | None,
    restrict_categories: Iterable[str],
    restrict_indices: NDArray[np.bool_],
    groups: NDArray,
) -> pd.Series[str]:
    key_added = f"{restrict_key}_R" if key_added is None else key_added
    all_groups = adata.obs[restrict_key].astype("U")
    prefix = f"{'-'.join(restrict_categories)},"
    new_groups = [prefix + g for g in groups.astype("U")]
    all_groups.iloc[restrict_indices] = new_groups
    return all_groups


def restrict_adjacency(
    adata: AnnData,
    restrict_key: str,
    *,
    restrict_categories: Sequence[str],
    adjacency: "CSBase",
) -> tuple["CSBase", NDArray[np.bool_]]:
    if not isinstance(restrict_categories[0], str):
        msg = "You need to use strings to label categories, e.g. '1' instead of 1."
        raise ValueError(msg)
    for c in restrict_categories:
        if c not in adata.obs[restrict_key].cat.categories:
            msg = f"{c!r} is not a valid category for {restrict_key!r}"
            raise ValueError(msg)
    restrict_indices = adata.obs[restrict_key].isin(restrict_categories).values
    adjacency = adjacency[restrict_indices, :]
    adjacency = adjacency[:, restrict_indices]
    return adjacency, restrict_indices


def get_igraph_from_adjacency(adjacency: "CSBase", *, directed: bool = False):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig

    sources, targets = adjacency.nonzero()
    weights = dematrix(adjacency[sources, targets]).ravel() if len(sources) else []
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])
    g.add_edges(list(zip(sources, targets, strict=True)))
    with suppress(KeyError):
        g.es["weight"] = weights
    if g.vcount() != adjacency.shape[0]:
        warnings.warn(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes.",
            UserWarning,
            stacklevel=2
        )
    return g


# ============================================================================
# Main Leiden Function
# ============================================================================

def leiden(  # noqa: PLR0912, PLR0913, PLR0915
    adata: AnnData,
    resolution: float = 1,
    *,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    random_state: int | np.random.RandomState = 0,
    key_added: str = "leiden",
    adjacency: "CSBase" | None = None,
    directed: bool | None = None,
    use_weights: bool = True,
    n_iterations: int | None = None,
    partition_type: type[MutableVertexPartition] | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
    flavor: Literal["brisc", "leidenalg", "leidenalg_exact", "igraph", None] = None,
    backend: str | None = None,
    **clustering_args,
) -> AnnData | None:
    """Cluster cells into subgroups :cite:p:`Traag2019`.

    Cluster cells using the Leiden algorithm :cite:p:`Traag2019`,
    an improved version of the Louvain algorithm :cite:p:`Blondel2008`.
    It was proposed for single-cell analysis by :cite:t:`Levine2015`.

    This requires having run :func:`~scanpy.pp.neighbors` or
    :func:`~scanpy.external.pp.bbknn` first.

    Parameters
    ----------
    adata
        The annotated data matrix.
    resolution
        A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters.
        Set to `None` if overriding `partition_type`
        to one that doesn't accept a `resolution_parameter`.
    random_state
        Change the initialization of the optimization.
    restrict_to
        Restrict the clustering to the categories within the key for sample
        annotation, tuple needs to contain `(obs_key, list_of_categories)`.
    key_added
        `adata.obs` key under which to add the cluster labels.
    adjacency
        Sparse adjacency matrix of the graph, defaults to neighbors connectivities.
    directed
        Whether to treat the graph as directed or undirected.
    use_weights
        If `True`, edge weights from the graph are used in the computation
        (placing more emphasis on stronger edges).
    n_iterations
        How many iterations of the Leiden clustering algorithm to perform.
        Positive values above 2 define the total number of iterations to perform,
        -1 has the algorithm run until it reaches its optimal clustering.
        Defaults to 2 with ``backend='cpp'`` and to -1 with ``backend='python'``.
        Not used by ``flavor='brisc'``, which runs one full Leiden pass
        (local moving, refinement and aggregation until no further merges).
    partition_type
        Type of partition to use.
        Defaults to :class:`~leidenalg.RBConfigurationVertexPartition`.
        For the available options, consult the documentation for
        :func:`~leidenalg.find_partition`.
    neighbors_key
        Use neighbors connectivities as adjacency.
        If not specified, leiden looks at .obsp['connectivities'] for connectivities
        (default storage place for pp.neighbors).
        If specified, leiden looks at
        .obsp[.uns[neighbors_key]['connectivities_key']] for connectivities.
    obsp
        Use .obsp[obsp] as adjacency. You can't specify both
        `obsp` and `neighbors_key` at the same time.
    copy
        Whether to copy `adata` or modify it inplace.
    flavor
        Which implementation to use. With ``backend='cpp'`` the default is
        ``'brisc'``: a C++ port of brisc's Leiden (same RB-configuration
        objective, ~4x faster than leidenalg, but different partitions);
        ``'leidenalg'``/``'leidenalg_exact'`` run an exact C++ port of
        leidenalg (identical partitions to the Python backend). With
        ``backend='python'`` the default is ``'leidenalg'``.
    backend
        ``'cpp'`` (default, :mod:`anndataoom_fast`) or ``'python'``; see
        ``ov.settings.backend``.
    **clustering_args
        Any further arguments to pass to :func:`~leidenalg.find_partition` (which in turn passes arguments to the `partition_type`)
        or :meth:`igraph.Graph.community_leiden` from `igraph`.

    Returns
    -------
    Returns `None` if `copy=False`, else returns an `AnnData` object. Sets the following fields:

    `adata.obs['leiden' | key_added]` : :class:`pandas.Series` (dtype ``category``)
        Array of dim (number of samples) that stores the subgroup id
        (``'0'``, ``'1'``, ...) for each cell.

    `adata.uns['leiden' | key_added]['params']` : :class:`dict`
        A dict with the values for the parameters `resolution`, `random_state`,
        and `n_iterations`.

    """
    from .._backend import resolve_backend
    _backend = resolve_backend(backend)
    if flavor is None and _backend == "cpp":
        # brisc's Leiden only optimises the RB-configuration objective; with a
        # custom partition type or extra leidenalg arguments keep leidenalg.
        if partition_type is None and not clustering_args:
            flavor = "brisc"
        else:
            flavor = "leidenalg"
    if n_iterations is None:
        n_iterations = 2 if _backend == "cpp" else -1
    if flavor == "brisc":
        if _backend != "cpp":
            raise ValueError("flavor='brisc' needs backend='cpp' (anndataoom-fast).")
        if partition_type is not None or clustering_args:
            raise ValueError("flavor='brisc' only supports the RB-configuration objective; "
                             "do not pass partition_type or extra clustering arguments.")
    exact = flavor == "leidenalg_exact"
    if exact:
        if _backend != "cpp":
            raise ValueError("flavor='leidenalg_exact' needs backend='cpp' (anndataoom-fast).")
        flavor = "leidenalg"
    if flavor is None:
        flavor = "leidenalg"
        msg = (
            "In the future, the default backend for leiden will be igraph instead of leidenalg. "
            "To achieve the future defaults please pass: `flavor='igraph'` and `n_iterations=2`. "
            "`directed` must also be `False` to work with igraph's implementation."
        )
        warn_once(msg, FutureWarning, stacklevel=2)
    if flavor not in {"igraph", "leidenalg", "brisc"}:
        msg = (
            "flavor must be 'brisc', 'leidenalg', 'leidenalg_exact' or 'igraph', "
            f"but {flavor!r} was passed"
        )
        raise ValueError(msg)
    if flavor != "brisc":
        ensure_igraph()
    if flavor == "brisc":
        pass
    elif flavor == "igraph":
        if directed:
            msg = "Cannot use igraph's leiden implementation with a directed graph."
            raise ValueError(msg)
        if partition_type is not None:
            msg = "Do not pass in partition_type argument when using igraph."
            raise ValueError(msg)
    else:
        try:
            import leidenalg
        except ImportError as e:
            msg = "Please install the leiden algorithm: `conda install -c conda-forge leidenalg` or `pip install leidenalg`."
            raise ImportError(msg) from e
    clustering_args = dict(clustering_args)

    start = info("running Leiden clustering")
    adata = adata.copy() if copy else adata
    # are we clustering a user-provided graph or the default AnnData one?
    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    if restrict_to is not None:
        restrict_key, restrict_categories = restrict_to
        adjacency, restrict_indices = restrict_adjacency(
            adata,
            restrict_key,
            restrict_categories=restrict_categories,
            adjacency=adjacency,
        )
    # Prepare find_partition arguments as a dictionary,
    # appending to whatever the user provided. It needs to be this way
    # as this allows for the accounting of a None resolution
    # (in the case of a partition variant that doesn't take it on input)
    clustering_args["n_iterations"] = n_iterations
    multi = isinstance(resolution, (list, tuple, np.ndarray))
    resolutions = [float(r) for r in resolution] if multi else [resolution]
    use_cpp = (
        _backend == "cpp"
        and flavor == "leidenalg"
        and (partition_type is None or partition_type is leidenalg.RBConfigurationVertexPartition)
        and set(clustering_args) <= {"n_iterations"}
        and isinstance(random_state, (int, np.integer))
        and all(r is not None for r in resolutions)
    )
    if flavor == "brisc":
        # C++ port of brisc's Leiden; resolutions run in parallel, labels do not
        # depend on the number of threads.
        import anndataoom_fast as _aof
        memberships = _aof.leiden_brisc_multi(
            adjacency, resolutions,
            random_state=int(random_state) if isinstance(random_state, (int, np.integer)) else 0)
    elif use_cpp:
        # Exact C++ port of leidenalg (same partition), several resolutions in parallel.
        import anndataoom_fast as _aof
        _directed = True if directed is None else directed
        memberships = _aof.leiden_multi(
            adjacency, resolutions, random_state=int(random_state), directed=_directed,
            n_iterations=n_iterations, use_weights=use_weights)
    else:
        memberships = []
        for _res in resolutions:
            _args = dict(clustering_args)
            if flavor == "leidenalg":
                if _res is not None:
                    _args["resolution_parameter"] = _res
                directed = True if directed is None else directed
                g = get_igraph_from_adjacency(adjacency, directed=directed)
                if partition_type is None:
                    partition_type = leidenalg.RBConfigurationVertexPartition
                if use_weights:
                    _args["weights"] = np.array(g.es["weight"]).astype(np.float64)
                _args["seed"] = random_state
                part = leidenalg.find_partition(g, partition_type, **_args)
            else:
                g = get_igraph_from_adjacency(adjacency, directed=False)
                if use_weights:
                    _args["weights"] = "weight"
                if _res is not None:
                    _args["resolution"] = _res
                _args.setdefault("objective_function", "modularity")
                with set_igraph_random_state(random_state):
                    part = g.community_leiden(**_args)
            memberships.append(np.array(part.membership))

    base_key = key_added
    for _res, groups in zip(resolutions, memberships):
        key_added = f"{base_key}_{_res}" if multi else base_key
        groups = np.asarray(groups)
        if restrict_to is not None:
            if key_added == "leiden":
                key_added += "_R"
            groups = rename_groups(
                adata,
                key_added=key_added,
                restrict_key=restrict_key,
                restrict_categories=restrict_categories,
                restrict_indices=restrict_indices,
                groups=groups,
            )
        labels = groups.astype("U")
        lab_np = np.asarray(labels)
        cats = natsorted(map(str, np.unique(lab_np)))
        try:
            adata.obs[key_added] = pd.Categorical(values=lab_np.astype("U"), categories=cats)
        except Exception:
            adata.obs[key_added] = labels
        if key_added != "leiden" and "leiden" not in adata.obs.columns:
            try:
                adata.obs["leiden"] = adata.obs[key_added]
            except Exception:
                adata.obs["leiden"] = pd.Categorical(values=lab_np.astype("U"), categories=cats)
        adata.uns[key_added] = {}
        adata.uns[key_added]["params"] = dict(
            resolution=_res,
            random_state=random_state,
            n_iterations=n_iterations,
        )
    info(
        "    finished",
        time=start,
        deep=(
            f"found {len(np.unique(groups))} clusters and added\n"
            f"    {key_added!r}, the cluster labels (adata.obs, categorical)"
        ),
    )
    return adata if copy else None
