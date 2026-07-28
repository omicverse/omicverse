"""Statistics of how spatial neighbours are arranged.

`ov.space` could already say *what is where* — domains, deconvolved cell types,
spatially variable genes. What it could not say is *how the things are arranged
relative to each other*: whether two cell types touch more often than chance,
how far apart their influence reaches, whether a gene changes as you walk away
from a boundary.

Every function here reads the graph :func:`omicverse.space.spatial_neighbors`
already builds — ``adata.obsp['spatial_connectivities']`` — so nothing has to be
recomputed, and results land in the ``adata.uns`` keys the ecosystem expects.

The semantics follow :mod:`squidpy`, deliberately: these statistics have one
correct definition each, and matching an existing reference implementation makes
the numbers checkable rather than merely plausible. ``tests/space/test_neighborhood.py``
holds every function against squidpy's own output on a shared fixture.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import sparse

from .._registry import register_function

__all__ = [
    "interaction_matrix",
    "nhood_enrichment",
    "centrality_scores",
    "co_occurrence",
    "ripley",
    "sepal",
    "mask_graph",
    "sliding_window",
    "var_by_distance",
]


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _get_graph(adata, connectivity_key: Optional[str]) -> sparse.csr_matrix:
    """Return the spatial connectivity graph, with a message that says what to run."""
    key = connectivity_key or "spatial_connectivities"
    if key not in adata.obsp:
        raise KeyError(
            f"No spatial graph at adata.obsp[{key!r}]. "
            "Build one first with `ov.space.spatial_neighbors(adata)`."
        )
    graph = adata.obsp[key]
    if not sparse.issparse(graph):
        graph = sparse.csr_matrix(graph)
    return graph.tocsr()


def _cluster_codes(adata, cluster_key: str) -> tuple[np.ndarray, list]:
    """Integer codes and ordered category names for a categorical obs column."""
    if cluster_key not in adata.obs:
        raise KeyError(f"adata.obs has no column {cluster_key!r}.")
    values = adata.obs[cluster_key]
    if not isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype("category")
    return np.asarray(values.cat.codes, dtype=np.int64), list(values.cat.categories)


def _coords(adata, spatial_key: str = "spatial") -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm has no key {spatial_key!r}.")
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm[{spatial_key!r}] must be n_obs x >=2.")
    return coords


def _counts_from_graph(graph: sparse.csr_matrix, codes: np.ndarray, n_cats: int,
                       weights: bool = False) -> np.ndarray:
    """Cluster-by-cluster edge counts.

    Every stored edge contributes to the (source cluster, target cluster) cell.
    The graph is symmetric, so each undirected edge is counted from both ends —
    the same convention squidpy uses, which is why the diagonal counts each
    within-cluster edge twice.
    """
    coo = graph.tocoo()
    valid = (codes[coo.row] >= 0) & (codes[coo.col] >= 0)
    rows = codes[coo.row[valid]]
    cols = codes[coo.col[valid]]
    vals = coo.data[valid].astype(np.float64) if weights else np.ones(valid.sum())
    out = np.zeros((n_cats, n_cats), dtype=np.float64)
    np.add.at(out, (rows, cols), vals)
    return out


# --------------------------------------------------------------------------- #
# interaction matrix
# --------------------------------------------------------------------------- #
@register_function(
    aliases=["交互矩阵", "interaction matrix", "簇间连接计数", "cluster interaction"],
    category="space",
    description="Count the graph edges running between each pair of clusters",
    requires={'obsp': ['spatial_connectivities'], 'obs': []},
    produces={'uns': ['{cluster_key}_interactions']},
    auto_fix='none',
    examples=[
        "ov.space.spatial_neighbors(adata)",
        "m = ov.space.interaction_matrix(adata, 'cell_type', copy=True)",
    ],
)
def interaction_matrix(
    adata,
    cluster_key: str,
    *,
    connectivity_key: Optional[str] = None,
    normalized: bool = False,
    weights: bool = False,
    copy: bool = False,
) -> Optional[np.ndarray]:
    """Count the edges running between every pair of clusters.

    This is the raw material the enrichment test is built on: how many times does
    a spot of type A sit next to a spot of type B. On its own it is dominated by
    how common each type is, which is what :func:`nhood_enrichment` corrects for.

    Arguments:
        adata: AnnData with a spatial graph and a categorical cluster column.
        cluster_key: Column in ``adata.obs`` holding the cluster labels.
        connectivity_key: Graph in ``adata.obsp``. Default ``'spatial_connectivities'``.
        normalized: Divide each row by its total, giving the fraction of a
            cluster's edges that land on each other cluster.
        weights: Use the stored edge weights instead of counting each edge as 1.
        copy: Return the matrix instead of writing it to ``adata.uns``.

    Returns:
        The ``n_clusters x n_clusters`` matrix when ``copy=True``, else ``None``.
    """
    graph = _get_graph(adata, connectivity_key)
    codes, cats = _cluster_codes(adata, cluster_key)
    out = _counts_from_graph(graph, codes, len(cats), weights=weights)

    if normalized:
        totals = out.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(totals > 0, out / totals, 0.0)

    if copy:
        return out
    adata.uns[f"{cluster_key}_interactions"] = out
    return None


# --------------------------------------------------------------------------- #
# neighbourhood enrichment
# --------------------------------------------------------------------------- #
@register_function(
    aliases=["邻域富集", "nhood enrichment", "neighborhood enrichment", "邻域富集分析", "空间共定位"],
    category="space",
    description="Permutation test for whether two clusters touch more or less often than chance",
    requires={'obsp': ['spatial_connectivities'], 'obs': []},
    produces={'uns': ['{cluster_key}_nhood_enrichment']},
    auto_fix='none',
    examples=[
        "ov.space.spatial_neighbors(adata)",
        "ov.space.nhood_enrichment(adata, 'cell_type', n_perms=1000, seed=0)",
    ],
)
def nhood_enrichment(
    adata,
    cluster_key: str,
    *,
    connectivity_key: Optional[str] = None,
    n_perms: int = 1000,
    seed: Optional[int] = None,
    copy: bool = False,
) -> Optional[dict]:
    """Test whether cluster pairs are adjacent more often than chance allows.

    The observed edge counts between clusters are compared against a null built
    by reshuffling the cluster labels over the fixed graph ``n_perms`` times. The
    z-score is the deviation of the observed count from that null, in null
    standard deviations, so it is not inflated by cluster size the way a raw
    count is.

    Shuffling labels — rather than rewiring the graph — holds tissue geometry
    constant and asks only about the labelling, which is the question worth
    asking.

    Arguments:
        adata: AnnData with a spatial graph and a categorical cluster column.
        cluster_key: Column in ``adata.obs`` holding the cluster labels.
        connectivity_key: Graph in ``adata.obsp``. Default ``'spatial_connectivities'``.
        n_perms: Number of label permutations forming the null. Default 1000.
        seed: Seed for the permutation RNG; set it if you want the z-scores back.
        copy: Return ``{'zscore', 'count'}`` instead of writing to ``adata.uns``.

    Returns:
        ``{'zscore': array, 'count': array}`` when ``copy=True``, else ``None``.

    Notes:
        A pair that never touches in any permutation has zero null variance; its
        z-score is reported as 0 rather than infinity.
    """
    if n_perms < 1:
        raise ValueError("n_perms must be >= 1.")

    graph = _get_graph(adata, connectivity_key)
    codes, cats = _cluster_codes(adata, cluster_key)
    n_cats = len(cats)

    observed = _counts_from_graph(graph, codes, n_cats)

    coo = graph.tocoo()
    rows, cols = coo.row, coo.col
    rng = np.random.default_rng(seed)

    perms = np.empty((n_perms, n_cats, n_cats), dtype=np.float64)
    shuffled = codes.copy()
    for k in range(n_perms):
        rng.shuffle(shuffled)
        r, c = shuffled[rows], shuffled[cols]
        valid = (r >= 0) & (c >= 0)
        mat = np.zeros((n_cats, n_cats), dtype=np.float64)
        np.add.at(mat, (r[valid], c[valid]), 1.0)
        perms[k] = mat

    mean = perms.mean(axis=0)
    std = perms.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        zscore = np.where(std > 0, (observed - mean) / std, 0.0)

    result = {"zscore": zscore, "count": observed.astype(np.int64)}
    if copy:
        return result
    adata.uns[f"{cluster_key}_nhood_enrichment"] = result
    return None


# --------------------------------------------------------------------------- #
# centrality
# --------------------------------------------------------------------------- #
@register_function(
    aliases=["中心性得分", "centrality scores", "簇中心性", "graph centrality"],
    category="space",
    description="Degree centrality, clustering coefficient and closeness of each cluster's induced subgraph",
    requires={'obsp': ['spatial_connectivities'], 'obs': []},
    produces={'uns': ['{cluster_key}_centrality_scores']},
    auto_fix='none',
    examples=[
        "ov.space.centrality_scores(adata, 'cell_type')",
        "df = ov.space.centrality_scores(adata, 'cell_type', copy=True)",
    ],
)
def centrality_scores(
    adata,
    cluster_key: str,
    *,
    score: Union[str, Iterable[str], None] = None,
    connectivity_key: Optional[str] = None,
    copy: bool = False,
) -> Optional[pd.DataFrame]:
    """Describe how each cluster sits inside the whole tissue graph.

    These are *group* centralities, measured for the cluster as a set of nodes
    within the full spatial graph — not properties of the cluster's own induced
    subgraph. The distinction matters: a cluster can be internally fragmented and
    still be highly central, because centrality is about its relationship to
    everything else.

    - ``degree_centrality`` — fraction of the spots outside the cluster that touch
      at least one spot inside it. High for a cluster that borders everything.
    - ``average_clustering`` — mean clustering coefficient of the cluster's spots
      in the full graph; high when the cluster forms a solid patch rather than a
      scatter of single spots.
    - ``closeness_centrality`` — inverse mean shortest-path distance from the
      cluster to the rest of the tissue. Low when the cluster is peripheral.

    Arguments:
        adata: AnnData with a spatial graph and a categorical cluster column.
        cluster_key: Column in ``adata.obs`` holding the cluster labels.
        score: Which scores to compute; default all three.
        connectivity_key: Graph in ``adata.obsp``. Default ``'spatial_connectivities'``.
        copy: Return the frame instead of writing to ``adata.uns``.

    Returns:
        A DataFrame indexed by cluster when ``copy=True``, else ``None``.
    """
    import networkx as nx

    all_scores = ["degree_centrality", "average_clustering", "closeness_centrality"]
    if score is None:
        wanted = all_scores
    else:
        wanted = [score] if isinstance(score, str) else list(score)
        unknown = set(wanted) - set(all_scores)
        if unknown:
            raise ValueError(f"Unknown score(s) {sorted(unknown)}; pick from {all_scores}.")

    graph = _get_graph(adata, connectivity_key)
    codes, cats = _cluster_codes(adata, cluster_key)
    whole = nx.from_scipy_sparse_array(graph)

    rows = []
    for i, cat in enumerate(cats):
        members = np.flatnonzero(codes == i)
        row: dict[str, Any] = {cluster_key: cat}
        if len(members) == 0:
            for s in wanted:
                row[s] = np.nan
            rows.append(row)
            continue
        idx = [int(m) for m in members]
        if "degree_centrality" in wanted:
            row["degree_centrality"] = float(
                nx.algorithms.centrality.group_degree_centrality(whole, idx)
            )
        if "average_clustering" in wanted:
            row["average_clustering"] = float(
                nx.algorithms.cluster.average_clustering(whole, nodes=idx)
            )
        if "closeness_centrality" in wanted:
            row["closeness_centrality"] = float(
                nx.algorithms.centrality.group_closeness_centrality(whole, idx)
            )
        rows.append(row)

    df = pd.DataFrame(rows).set_index(cluster_key)
    if copy:
        return df
    adata.uns[f"{cluster_key}_centrality_scores"] = df
    return None


# --------------------------------------------------------------------------- #
# co-occurrence
# --------------------------------------------------------------------------- #
@register_function(
    aliases=["共现概率", "co-occurrence", "空间共现", "distance co-occurrence"],
    category="space",
    description="Co-occurrence probability of cluster pairs as a function of distance",
    requires={'obsm': ['spatial'], 'obs': []},
    produces={'uns': ['{cluster_key}_co_occurrence']},
    auto_fix='none',
    examples=[
        "ov.space.co_occurrence(adata, 'cell_type', interval=50)",
        "occ, dist = ov.space.co_occurrence(adata, 'cell_type', copy=True)",
    ],
)
def co_occurrence(
    adata,
    cluster_key: str,
    *,
    spatial_key: str = "spatial",
    interval: Union[int, np.ndarray] = 50,
    copy: bool = False,
):
    """How the odds of finding cluster B near cluster A change with distance.

    For every radius the statistic is the conditional probability of seeing
    cluster ``B`` within that radius of a cluster-``A`` spot, divided by the
    unconditional probability of ``B`` anywhere. Above 1 means enrichment, and
    the distance at which it falls back to 1 is the range over which the
    association holds.

    Unlike :func:`nhood_enrichment` this ignores the graph entirely and works
    from raw distances, so it answers "how far", not just "adjacent or not".

    Arguments:
        adata: AnnData with spatial coordinates and a categorical cluster column.
        cluster_key: Column in ``adata.obs`` holding the cluster labels.
        spatial_key: Coordinates in ``adata.obsm``. Default ``'spatial'``.
        interval: Number of distance thresholds, or the thresholds themselves.
        copy: Return ``(occurrence, intervals)`` instead of writing to ``adata.uns``.

    Returns:
        ``(occurrence, intervals)`` when ``copy=True``, else ``None``. ``occurrence``
        has shape ``(n_clusters, n_clusters, n_intervals)`` and is indexed
        ``[condition, target, step]``.
    """
    from scipy.spatial import distance_matrix

    coords = _coords(adata, spatial_key)
    codes, cats = _cluster_codes(adata, cluster_key)
    n_cats = len(cats)

    dist = distance_matrix(coords, coords)

    if np.isscalar(interval):
        # squidpy's bracket: shortest edge between the two lowest-sum corners,
        # to half the diagonal of the tissue
        coord_sum = coords.sum(axis=1)
        lo1, lo2 = np.argpartition(coord_sum, 2)[:2]
        hi = int(np.argmax(coord_sum))
        thresh_min = float(np.linalg.norm(coords[lo1] - coords[lo2]))
        thresh_max = float(np.linalg.norm(coords[lo1] - coords[hi]) / 2.0)
        edges = np.linspace(thresh_min, thresh_max, num=int(interval), dtype=np.float64)
    else:
        edges = np.array(sorted(np.asarray(interval, dtype=np.float64)), copy=True)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("interval must be an int >= 2 or a 1-D array of at least two radii.")

    # the radii are edges: n edges give n-1 cumulative steps, each using the
    # upper edge as the radius
    n_steps = len(edges) - 1
    occurrence = np.zeros((n_cats, n_cats, n_steps), dtype=np.float64)

    for s in range(n_steps):
        radius = edges[s + 1]
        within = (dist <= radius) & (dist > 0)
        co = np.zeros((n_cats, n_cats), dtype=np.float64)
        rows, cols = np.nonzero(within)
        valid = (codes[rows] >= 0) & (codes[cols] >= 0)
        np.add.at(co, (codes[rows[valid]], codes[cols[valid]]), 1.0)

        total = co.sum()
        # the marginal comes from the pairs inside this radius, not from the
        # global cluster frequencies — a spot in a sparse region contributes
        # fewer pairs and so weighs less
        probs = co.sum(axis=0) / total if total else np.zeros(n_cats)
        for c in range(n_cats):
            row_sum = co[c].sum()
            cond = co[c] / row_sum if row_sum else np.zeros(n_cats)
            with np.errstate(invalid="ignore", divide="ignore"):
                occurrence[c, :, s] = np.where(probs > 0, cond / probs, 0.0)

    if copy:
        return occurrence, edges
    adata.uns[f"{cluster_key}_co_occurrence"] = {"occ": occurrence, "interval": edges}
    return None


# --------------------------------------------------------------------------- #
# Ripley
# --------------------------------------------------------------------------- #
@register_function(
    aliases=["Ripley统计量", "ripley", "点模式分析", "spatial point pattern"],
    category="space",
    description="Ripley's F, G or L statistic per cluster against a random-point null",
    requires={'obsm': ['spatial'], 'obs': []},
    produces={'uns': ['{cluster_key}_ripley_{mode}']},
    auto_fix='none',
    examples=[
        "ov.space.ripley(adata, 'cell_type', mode='L')",
        "res = ov.space.ripley(adata, 'cell_type', mode='G', copy=True)",
    ],
)
def ripley(
    adata,
    cluster_key: str,
    *,
    mode: str = "F",
    spatial_key: str = "spatial",
    n_neigh: int = 2,
    n_simulations: int = 100,
    n_observations: int = 1000,
    max_dist: Optional[float] = None,
    n_steps: int = 50,
    seed: Optional[int] = None,
    copy: bool = False,
):
    """Ask whether a cluster's spots are clustered, dispersed, or just random.

    Three classical point-pattern statistics, each answering the question from a
    different side:

    - ``'F'`` empty-space function — from random probe points, distance to the
      nearest cluster spot. Sensitive to gaps.
    - ``'G'`` nearest-neighbour function — from each cluster spot, distance to
      the next one. Sensitive to tight packing.
    - ``'L'`` Besag's transform of Ripley's K — variance-stabilised count of
      neighbours within a radius. Above the null means clustered, below means
      dispersed.

    The null is drawn by scattering the same number of points uniformly over the
    bounding box ``n_simulations`` times, so "random" means random *given the
    tissue's extent*.

    Arguments:
        adata: AnnData with spatial coordinates and a categorical cluster column.
        cluster_key: Column in ``adata.obs`` holding the cluster labels.
        mode: ``'F'``, ``'G'`` or ``'L'``.
        spatial_key: Coordinates in ``adata.obsm``. Default ``'spatial'``.
        n_neigh: Neighbour rank used by the ``'F'`` and ``'G'`` modes. Default 2.
        n_simulations: Null replicates. Default 100.
        n_observations: Probe points per replicate (``'F'`` mode). Default 1000.
        max_dist: Largest radius; defaults to a quarter of the bounding-box diagonal.
        n_steps: Radii sampled between 0 and ``max_dist``. Default 50.
        seed: RNG seed.
        copy: Return the result dict instead of writing to ``adata.uns``.

    Returns:
        ``{'{mode}_stat': DataFrame, 'sims_stat': DataFrame, 'bins': array}`` when
        ``copy=True``, else ``None``.
    """
    from matplotlib.path import Path as _Path
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import pdist
    from sklearn.neighbors import NearestNeighbors

    mode = str(mode).upper()
    if mode not in ("F", "G", "L"):
        raise ValueError("mode must be one of 'F', 'G', 'L'.")

    coords = _coords(adata, spatial_key)
    codes, cats = _cluster_codes(adata, cluster_key)
    n_obs_total = coords.shape[0]

    hull = ConvexHull(coords)
    area = float(hull.volume)          # 'volume' is the enclosed area in 2-D
    if max_dist is None:
        max_dist = float((area / 2.0) ** 0.5)
    support = np.linspace(0.0, max_dist, n_steps)

    hull_path = _Path(coords[hull.vertices])
    lo, hi = coords.min(axis=0), coords.max(axis=0)

    def _poisson_points(n: int, rng) -> np.ndarray:
        """Uniform points inside the tissue outline, by rejection in its bounding box."""
        out = np.empty((0, coords.shape[1]), dtype=np.float64)
        while len(out) < n:
            draw = rng.uniform(lo, hi, size=(max(n * 2, 64), coords.shape[1]))
            out = np.vstack([out, draw[hull_path.contains_points(draw[:, :2])]])
        return out[:n]

    def _f_g(distances: np.ndarray) -> np.ndarray:
        """Cumulative empirical distribution of distances over the support."""
        counts, _ = np.histogram(distances, bins=support)
        total = counts.sum()
        fracs = np.cumsum(counts) / total if total else np.zeros_like(counts, dtype=float)
        return np.concatenate(([0.0], fracs))

    def _l(distances: np.ndarray) -> np.ndarray:
        """Besag's L, with intensity taken over all observations and the hull area."""
        pairs = (distances < support.reshape(-1, 1)).sum(axis=1)
        intensity = n_obs_total / area if area > 0 else 0.0
        k = ((pairs * 2) / n_obs_total) / intensity if intensity > 0 else np.zeros_like(support)
        return np.sqrt(np.clip(k, 0, None) / np.pi)

    obs_rows = []
    probe = None
    for i, cat in enumerate(cats):
        pts = coords[codes == i]
        if len(pts) < max(n_neigh, 2):
            stats = np.zeros_like(support)
        elif mode == "F":
            rng = np.random.default_rng(seed)
            probe = _poisson_points(n_observations, rng)
            nn = NearestNeighbors(n_neighbors=n_neigh).fit(pts)
            d, _ = nn.kneighbors(probe, n_neighbors=n_neigh)
            stats = _f_g(np.squeeze(d))
        elif mode == "G":
            # distance from every spot *outside* the cluster to the cluster
            others = coords[codes != i]
            if len(others) == 0:
                stats = np.zeros_like(support)
            else:
                nn = NearestNeighbors(n_neighbors=n_neigh).fit(pts)
                d, _ = nn.kneighbors(others, n_neighbors=n_neigh)
                stats = _f_g(np.squeeze(d))
        else:
            stats = _l(pdist(pts))
        obs_rows.append(pd.DataFrame({"bins": support, "stats": stats, cluster_key: cat}))

    sim_rows = []
    for s in range(n_simulations):
        rng = np.random.default_rng(None if seed is None else seed + s)
        rand = _poisson_points(n_observations, rng)
        if mode == "F":
            if probe is None:
                probe = _poisson_points(n_observations, np.random.default_rng(seed))
            nn = NearestNeighbors(n_neighbors=n_neigh).fit(rand)
            d, _ = nn.kneighbors(probe, n_neighbors=1)
            stats = _f_g(np.squeeze(d))
        elif mode == "G":
            nn = NearestNeighbors(n_neighbors=n_neigh).fit(rand)
            d, _ = nn.kneighbors(coords, n_neighbors=1)
            stats = _f_g(np.squeeze(d))
        else:
            stats = _l(pdist(rand))
        sim_rows.append(pd.DataFrame({"bins": support, "stats": stats, "sim": s}))

    result = {
        f"{mode}_stat": pd.concat(obs_rows, ignore_index=True),
        "sims_stat": pd.concat(sim_rows, ignore_index=True),
        "bins": support,
    }
    if copy:
        return result
    adata.uns[f"{cluster_key}_ripley_{mode}"] = result
    return None


# --------------------------------------------------------------------------- #
# sepal
# --------------------------------------------------------------------------- #
@register_function(
    aliases=["sepal", "扩散空间基因", "diffusion spatial genes", "空间结构基因"],
    category="space",
    description="Rank genes by how long their spatial pattern survives diffusion (sepal)",
    requires={'obsm': ['spatial'], 'obsp': ['spatial_connectivities']},
    produces={'uns': ['sepal_score']},
    auto_fix='none',
    examples=[
        "ov.space.spatial_neighbors(adata, n_neighs=6)",
        "ov.space.sepal(adata, max_neighs=6, genes=adata.var_names[:200])",
    ],
)
def sepal(
    adata,
    max_neighs: int,
    *,
    genes: Union[str, Sequence[str], None] = None,
    n_iter: int = 30000,
    dt: float = 0.001,
    thresh: float = 1e-8,
    connectivity_key: str = "spatial_connectivities",
    spatial_key: str = "spatial",
    layer: Optional[str] = None,
    use_raw: bool = False,
    copy: bool = False,
) -> Optional[pd.DataFrame]:
    """Score genes by how long their spatial structure resists diffusion.

    Let each gene's expression diffuse over the spatial lattice and record the
    time it takes to flatten. A gene with real spatial organisation takes longer
    than one scattered at random, and that time is the score. Unlike a variance
    test this rewards *structure* rather than magnitude, so a modestly expressed
    gene confined to one region can outrank a loud but diffuse one.

    Requires a regular lattice — ``max_neighs=6`` for Visium's hexagonal grid,
    ``4`` for a square grid — because the diffusion operator assumes each
    interior spot has the same number of neighbours.

    Arguments:
        adata: AnnData with coordinates and a spatial graph.
        max_neighs: 4 for a square lattice, 6 for a hexagonal one.
        genes: Genes to score; default all of ``adata.var_names``.
        n_iter: Maximum diffusion steps before giving up. Default 30000.
        dt: Time step. Default 0.001.
        thresh: Convergence threshold on the gradient norm. Default 1e-8.
        connectivity_key: Graph in ``adata.obsp``.
        spatial_key: Coordinates in ``adata.obsm``.
        layer: Layer to read expression from; default ``adata.X``.
        use_raw: Read from ``adata.raw`` instead.
        copy: Return the scores instead of writing to ``adata.uns``.

    Returns:
        A DataFrame of scores indexed by gene when ``copy=True``, else ``None``.
    """
    if max_neighs not in (4, 6):
        raise ValueError("max_neighs must be 4 (square lattice) or 6 (hexagonal).")

    graph = _get_graph(adata, connectivity_key)
    _coords(adata, spatial_key)

    if use_raw:
        source = adata.raw
        var_names = list(source.var_names)
        matrix = source.X
    elif layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"adata.layers has no key {layer!r}.")
        var_names = list(adata.var_names)
        matrix = adata.layers[layer]
    else:
        var_names = list(adata.var_names)
        matrix = adata.X

    if genes is None:
        wanted = var_names
    else:
        wanted = [genes] if isinstance(genes, str) else list(genes)
        missing = set(wanted) - set(var_names)
        if missing:
            raise KeyError(f"{len(missing)} gene(s) not found, e.g. {sorted(missing)[:3]}")

    idx = [var_names.index(g) for g in wanted]
    dense = matrix[:, idx]
    dense = np.asarray(dense.todense() if sparse.issparse(dense) else dense, dtype=np.float64)

    # A spot is *saturated* when it has the full neighbour complement, i.e. it is
    # in the interior of the lattice. Diffusion is solved on those; the rim spots
    # simply follow their nearest saturated neighbour, which keeps the boundary
    # from acting as a sink and draining the tissue.
    graph = graph.tocsr()
    degree = np.diff(graph.indptr)
    sat = np.flatnonzero(degree == max_neighs)
    unsat = np.flatnonzero(degree < max_neighs)
    if sat.size == 0:
        raise ValueError(
            f"No spot has exactly {max_neighs} neighbours — the graph is not the "
            f"lattice this score assumes. Rebuild with "
            f"`ov.space.spatial_neighbors(adata, n_neighs={max_neighs})`."
        )

    sat_idx = np.vstack([graph.indices[graph.indptr[i]:graph.indptr[i + 1]] for i in sat])

    # each rim spot borrows from a saturated neighbour where it has one, and
    # otherwise from the nearest saturated spot in space
    from sklearn.metrics import pairwise_distances

    sat_set = set(sat.tolist())
    nearest_sat = np.full(unsat.shape[0], -1, dtype=np.int64)
    leftover = []
    for pos, i in enumerate(unsat):
        nb = graph.indices[graph.indptr[i]:graph.indptr[i + 1]]
        hit = [n for n in nb if n in sat_set]
        if hit:
            nearest_sat[pos] = hit[0]
        else:
            leftover.append(pos)
    if leftover:
        coords_all = _coords(adata, spatial_key)
        d = pairwise_distances(coords_all[unsat[leftover]], coords_all[sat], metric="l1")
        nearest_sat[leftover] = sat[np.argmin(d, axis=1)]

    stencil_centre = float(max_neighs)
    hex_factor = 2.0 / 3.0 if max_neighs == 6 else 1.0

    def _entropy(x: np.ndarray) -> float:
        nz = x[x > 0]
        total = nz.sum()
        if total <= 0:
            return 0.0
        p = nz / total
        return float(-(np.log(p) * p).sum())

    scores = np.zeros(len(wanted), dtype=np.float64)
    for j in range(dense.shape[1]):
        conc = dense[:, j].astype(np.float64).copy()
        prev_ent = 1.0
        hit_iter = np.nan
        for it in range(n_iter):
            nhood = conc[sat_idx].sum(axis=1)
            d2 = (nhood - stencil_centre * conc[sat]) * hex_factor
            dcdt = np.zeros_like(conc)
            dcdt[sat] = d2
            conc[sat] += dcdt[sat] * dt
            conc[unsat] += dcdt[nearest_sat] * dt
            conc[conc < 0] = 0.0
            ent = _entropy(conc[sat]) / sat.shape[0]
            if abs(ent - prev_ent) <= thresh:
                hit_iter = it
                break
            prev_ent = ent
        scores[j] = dt * hit_iter

    df = pd.DataFrame({"sepal_score": scores}, index=wanted).sort_values(
        "sepal_score", ascending=False
    )
    if copy:
        return df
    adata.uns["sepal_score"] = df
    return None


# --------------------------------------------------------------------------- #
# mask
# --------------------------------------------------------------------------- #
@register_function(
    aliases=["图掩膜", "mask graph", "区域裁剪图", "polygon mask"],
    category="space",
    description="Restrict a spatial graph to the observations inside (or outside) a polygon",
    requires={'obsm': ['spatial'], 'obsp': ['spatial_connectivities']},
    produces={'obs': ['{key_added}'], 'obsp': ['{key_added}_connectivities']},
    auto_fix='none',
    examples=[
        "poly = [(0, 0), (0, 500), (500, 500), (500, 0)]",
        "ov.space.mask_graph(adata, poly, key_added='roi')",
    ],
)
def mask_graph(
    adata,
    polygon,
    *,
    negative_mask: bool = False,
    spatial_key: str = "spatial",
    connectivity_key: Optional[str] = None,
    key_added: str = "mask",
    copy: bool = False,
):
    """Keep only the part of the spatial graph inside a hand-drawn region.

    Every statistic in this module is computed over whichever graph it is given,
    so restricting the graph is how you restrict an analysis to one region —
    a tumour bed, a cortical layer — without subsetting the object and losing the
    surrounding context.

    Arguments:
        adata: AnnData with coordinates and a spatial graph.
        polygon: Vertices as an ``(n, 2)`` array, a list of ``(x, y)`` pairs, or a
            shapely ``Polygon``/``MultiPolygon``.
        negative_mask: Keep what is *outside* the polygon instead.
        spatial_key: Coordinates in ``adata.obsm``.
        connectivity_key: Graph in ``adata.obsp``. Default ``'spatial_connectivities'``.
        key_added: Prefix for the boolean ``obs`` column and the masked graph.
        copy: Return ``(mask, graph)`` instead of writing to ``adata``.

    Returns:
        ``(mask, masked_graph)`` when ``copy=True``, else ``None``.
    """
    from matplotlib.path import Path as _Path

    coords = _coords(adata, spatial_key)
    graph = _get_graph(adata, connectivity_key)

    if hasattr(polygon, "geoms"):                       # shapely MultiPolygon
        inside = np.zeros(len(coords), dtype=bool)
        for geom in polygon.geoms:
            inside |= _Path(np.asarray(geom.exterior.coords)).contains_points(coords)
    elif hasattr(polygon, "exterior"):                  # shapely Polygon
        inside = _Path(np.asarray(polygon.exterior.coords)).contains_points(coords)
    else:
        verts = np.asarray(polygon, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError("polygon must be an (n, 2) array of vertices.")
        inside = _Path(verts).contains_points(coords)

    keep = ~inside if negative_mask else inside
    diag = sparse.diags(keep.astype(np.float64))
    masked = (diag @ graph @ diag).tocsr()
    masked.eliminate_zeros()

    if copy:
        return keep, masked
    adata.obs[key_added] = keep
    adata.obsp[f"{key_added}_connectivities"] = masked
    return None


# --------------------------------------------------------------------------- #
# sliding window
# --------------------------------------------------------------------------- #
@register_function(
    aliases=["滑动窗口", "sliding window", "空间分窗", "tile assignment"],
    category="space",
    description="Assign every observation to a square sliding window over the tissue",
    requires={'obsm': ['spatial']},
    produces={'obs': ['{sliding_window_key}']},
    auto_fix='none',
    examples=[
        "ov.space.sliding_window(adata, window_size=500)",
        "ov.space.sliding_window(adata, window_size=500, overlap=250)",
    ],
)
def sliding_window(
    adata,
    *,
    window_size: Optional[float] = None,
    overlap: float = 0,
    spatial_key: str = "spatial",
    library_key: Optional[str] = None,
    sliding_window_key: str = "sliding_window_assignment",
    drop_partial_windows: bool = False,
    copy: bool = False,
):
    """Tile the tissue into square windows and label every spot with its window.

    Useful for two things: turning a slide into pseudo-replicates so a statistic
    can be given an error bar, and testing whether a result holds locally rather
    than only over the whole section.

    With ``overlap`` above zero the windows are laid down every
    ``window_size - overlap`` units, so a spot can fall in several; the assignment
    then names the first window that contains it.

    Arguments:
        adata: AnnData with spatial coordinates.
        window_size: Side length in coordinate units. Defaults to a quarter of the
            smaller extent of the tissue.
        overlap: How much neighbouring windows share. Must be < ``window_size``.
        spatial_key: Coordinates in ``adata.obsm``.
        library_key: Tile each library separately when given.
        sliding_window_key: Name of the ``obs`` column to write.
        drop_partial_windows: Leave spots in windows that hang off the edge
            unassigned instead of keeping them.
        copy: Return the assignment instead of writing it to ``adata.obs``.

    Returns:
        A Series of window labels when ``copy=True``, else ``None``.
    """
    coords = _coords(adata, spatial_key)
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")

    lo, hi = coords.min(axis=0), coords.max(axis=0)
    if window_size is None:
        window_size = float(np.min(hi[:2] - lo[:2]) / 4.0)
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size.")

    step = window_size - overlap
    libs = (adata.obs[library_key].astype(str).to_numpy()
            if library_key is not None else np.zeros(len(coords), dtype=int).astype(str))

    labels = np.full(len(coords), None, dtype=object)
    for lib in pd.unique(libs):
        sel = libs == lib
        pts = coords[sel][:, :2]
        p_lo, p_hi = pts.min(axis=0), pts.max(axis=0)
        xs = np.arange(p_lo[0], p_hi[0] + step, step)
        ys = np.arange(p_lo[1], p_hi[1] + step, step)
        sub = np.full(sel.sum(), None, dtype=object)
        for wi, x0 in enumerate(xs):
            for wj, y0 in enumerate(ys):
                if drop_partial_windows and (x0 + window_size > p_hi[0] or y0 + window_size > p_hi[1]):
                    continue
                inside = ((pts[:, 0] >= x0) & (pts[:, 0] < x0 + window_size)
                          & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + window_size))
                fresh = inside & np.equal(sub, None)
                if fresh.any():
                    name = f"window_{wi}_{wj}" if library_key is None else f"{lib}_window_{wi}_{wj}"
                    sub[fresh] = name
        labels[sel] = sub

    series = pd.Series(pd.Categorical(labels), index=adata.obs_names,
                       name=sliding_window_key)
    if copy:
        return series
    adata.obs[sliding_window_key] = series
    return None


# --------------------------------------------------------------------------- #
# variables by distance
# --------------------------------------------------------------------------- #
@register_function(
    aliases=["距离依赖表达", "var by distance", "梯度分析", "distance gradient"],
    category="space",
    description="Build a design matrix of each observation's distance to an anchor cluster",
    requires={'obsm': ['spatial'], 'obs': []},
    produces={'obsm': ['{design_matrix_key}']},
    auto_fix='none',
    examples=[
        "ov.space.var_by_distance(adata, groups='tumour', cluster_key='cell_type')",
        "dm = adata.obsm['design_matrix']",
    ],
)
def var_by_distance(
    adata,
    groups: Union[str, Sequence[str], np.ndarray],
    *,
    cluster_key: Optional[str] = None,
    spatial_key: str = "spatial",
    design_matrix_key: str = "design_matrix",
    covariates: Union[str, Sequence[str], None] = None,
    metric: str = "euclidean",
    library_key: Optional[str] = None,
    normalize: bool = True,
    copy: bool = False,
) -> Optional[pd.DataFrame]:
    """Measure every spot's distance to an anchor, so expression can be read against it.

    The anchor is one or more clusters — a tumour core, a vessel, a wound edge.
    The result is a table of distances that turns "is this gene spatially
    variable" into the sharper question "does this gene change as you move away
    from *that*", which is usually the one being asked.

    Arguments:
        adata: AnnData with spatial coordinates.
        groups: Anchor cluster name(s), or an ``(n, 2)`` array of anchor coordinates.
        cluster_key: Column holding the cluster labels; required when ``groups``
            names clusters.
        spatial_key: Coordinates in ``adata.obsm``.
        design_matrix_key: Key in ``adata.obsm`` to write.
        covariates: Extra ``obs`` columns to carry into the design matrix.
        metric: Any metric ``scipy.spatial.distance.cdist`` accepts.
        library_key: Compute distances within each library separately.
        normalize: Scale each anchor's distances to ``[0, 1]`` by their maximum, so
            slides of different physical size are comparable. Set ``False`` to keep
            the distances in the coordinate units they came in.
        copy: Return the design matrix instead of writing it to ``adata.obsm``.

    Returns:
        The design matrix when ``copy=True``, else ``None``.
    """
    from scipy.spatial.distance import cdist

    coords = _coords(adata, spatial_key)

    anchor_coords: Optional[np.ndarray] = None
    anchor_names: list[str] = []
    if isinstance(groups, np.ndarray) and groups.ndim == 2:
        anchor_coords = np.asarray(groups, dtype=np.float64)
        anchor_names = ["anchor"]
    else:
        if cluster_key is None:
            raise ValueError("cluster_key is required when `groups` names clusters.")
        if cluster_key not in adata.obs:
            raise KeyError(f"adata.obs has no column {cluster_key!r}.")
        anchor_names = [groups] if isinstance(groups, str) else list(groups)
        labels = adata.obs[cluster_key].astype(str).to_numpy()
        unknown = set(anchor_names) - set(np.unique(labels))
        if unknown:
            raise ValueError(f"{sorted(unknown)} not found in adata.obs[{cluster_key!r}].")

    frame = pd.DataFrame(index=adata.obs_names)
    libs = (adata.obs[library_key].astype(str).to_numpy()
            if library_key is not None else np.zeros(len(coords), dtype=int).astype(str))

    for name in anchor_names:
        dist = np.full(len(coords), np.nan, dtype=np.float64)
        for lib in pd.unique(libs):
            sel = libs == lib
            if anchor_coords is not None:
                target = anchor_coords
            else:
                target = coords[sel][labels[sel] == name]
            if len(target) == 0:
                continue
            dist[sel] = cdist(coords[sel], target, metric=metric).min(axis=1)

        if normalize:
            frame[f"{name}_raw"] = dist.copy()
            scaled = dist.astype(np.float64).copy()
            # an anchor spot is at distance 0 from itself; that is not a
            # measurement of anything, so it drops out rather than anchoring the
            # scale at zero
            scaled[scaled == 0] = np.nan
            if np.isfinite(scaled).any():
                smallest = np.nanargmin(scaled)
                scaled[smallest] = 0.0
                span = np.nanmax(scaled) - np.nanmin(scaled)
                if span > 0:
                    scaled = (scaled - np.nanmin(scaled)) / span
            frame[name] = scaled
        else:
            frame[name] = dist

    if covariates is not None:
        for cov in ([covariates] if isinstance(covariates, str) else list(covariates)):
            if cov not in adata.obs:
                raise KeyError(f"adata.obs has no covariate column {cov!r}.")
            frame[cov] = adata.obs[cov].to_numpy()

    if library_key is not None:
        frame[library_key] = adata.obs[library_key].to_numpy()

    if copy:
        return frame
    adata.obsm[design_matrix_key] = frame
    return None
