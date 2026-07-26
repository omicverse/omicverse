"""
Cell ordering and pseudotime assignment for monocle2_py.

Implements orderCells() — assigns pseudotime and state to each cell.
"""

import numpy as np
import pandas as pd
import igraph as ig
from scipy.spatial.distance import pdist, squareform


def _euclidean_mst_delaunay(pts, N_cells, _knn_k_start=50):
    """Compute the exact Euclidean MST of a point cloud.

    Strategy: Euclidean MST is a subgraph of the Delaunay triangulation
    in any dimension (Preparata & Shamos 1985), so we only need to run
    scipy MST on the Delaunay edge set.  This is O(N·d) memory instead
    of O(N²).

    Matches R Monocle 2's project2MST::

        dp <- as.matrix(dist(t(P)))
        dp <- dp + min(dp[dp != 0]); diag(dp) <- 0
        minimum.spanning.tree(graph.adjacency(dp, weighted=TRUE))

    The ``+ min_dist`` shift is applied to every edge so pseudotime
    values agree with R bitwise (the shift leaves MST *topology*
    unchanged but changes cumulative path lengths).

    Fallback: if scipy.spatial.Delaunay raises ``QhullError`` (e.g.
    all-coplanar points), we fall back to a k-NN graph with
    ``.maximum(.T)`` symmetrisation and adaptive k.

    Parameters
    ----------
    pts : ndarray, (N, d)
    N_cells : int
        Must equal ``pts.shape[0]``.
    _knn_k_start : int
        Initial k for the kNN fallback (doubled until the graph is
        connected).

    Returns
    -------
    (mst_coo, projected_dp_sparse) :
        * ``mst_coo`` — scipy sparse COO of the MST (N-1 edges,
          directed: one entry per edge).
        * ``projected_dp_sparse`` — scipy CSR matrix of SYMMETRIC MST
          edges; ``_extract_ddrtree_ordering`` looks up MST edge
          distances here.
    """
    from scipy.spatial import Delaunay
    try:
        from scipy.spatial import QhullError           # scipy ≥ 1.8
    except ImportError:                                # pragma: no cover
        from scipy.spatial.qhull import QhullError     # scipy < 1.8
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import (
        minimum_spanning_tree, connected_components,
    )

    if N_cells <= 1:
        # Degenerate: 0 or 1 cell → empty MST
        return csr_matrix((N_cells, N_cells)).tocoo(), csr_matrix((N_cells, N_cells))

    try:
        # ``QJ`` = "joggled input" perturbation to handle near-degenerate
        # configurations (co-linear or co-planar points). This is the
        # recommended qhull option for robust triangulation of
        # single-cell trajectory data, which often sits on a low-dim
        # manifold.
        tri = Delaunay(pts, qhull_options='QJ')
        simplices = tri.simplices                 # (M, d+1)
        # Enumerate all unique pairs (i, j) within each simplex.
        # For a simplex with d+1 vertices there are (d+1)*d/2 pairs.
        r = []
        c = []
        for s in simplices:
            s_sorted = np.sort(s)
            for ii in range(len(s_sorted)):
                for jj in range(ii + 1, len(s_sorted)):
                    r.append(s_sorted[ii])
                    c.append(s_sorted[jj])
        r = np.asarray(r, dtype=np.int64)
        c = np.asarray(c, dtype=np.int64)
        # Deduplicate via 1-D encoding
        key = r * N_cells + c
        _, uniq = np.unique(key, return_index=True)
        r, c = r[uniq], c[uniq]

        raw_d = np.linalg.norm(pts[r] - pts[c], axis=1)
        source = 'delaunay'
    except (QhullError, Exception) as _err:
        # Fallback: k-NN graph with `.maximum(.T)` symmetrisation.
        # Adaptive k: double until the graph is connected.
        import warnings as _w
        _w.warn(
            f"Delaunay triangulation failed ({_err!r}); falling back "
            "to k-NN MST. Results may differ slightly from the exact "
            "Euclidean MST for near-degenerate point clouds.",
            RuntimeWarning,
        )
        from sklearn.neighbors import kneighbors_graph as _knng
        k = min(_knn_k_start, N_cells - 1)
        A = None
        while k < N_cells:
            A = _knng(pts, n_neighbors=k, mode='distance',
                       include_self=False)
            # Symmetrise with maximum: keeps the real distance when one
            # direction is missing (sparse missing = 0; min would wipe
            # the real value, max preserves it).
            A = A.maximum(A.T)
            n_comp, _lbl = connected_components(A, directed=False,
                                                  return_labels=True)
            if n_comp == 1:
                break
            k = min(k * 2, N_cells - 1)
        if A is None:
            raise RuntimeError("k-NN MST fallback failed: N too small")
        coo = A.tocoo()
        # Keep only the upper triangle to mirror Delaunay path
        mask = coo.row < coo.col
        r = coo.row[mask]
        c = coo.col[mask]
        raw_d = coo.data[mask]
        source = 'knn'

    if raw_d.size == 0:
        # Degenerate case — single cell
        return (csr_matrix((N_cells, N_cells)).tocoo(),
                csr_matrix((N_cells, N_cells)))

    # R's constant min_dist shift (preserves topology, changes weights)
    min_dist = float(raw_d[raw_d > 0].min()) if (raw_d > 0).any() else 1e-10
    weights = raw_d + min_dist

    # Build symmetric sparse graph for scipy MST
    all_r = np.concatenate([r, c])
    all_c = np.concatenate([c, r])
    all_w = np.concatenate([weights, weights])
    sym_sparse = csr_matrix((all_w, (all_r, all_c)),
                             shape=(N_cells, N_cells))

    mst = minimum_spanning_tree(sym_sparse).tocoo()

    # `projected_dp` must be symmetric (the pseudotime lookup
    # ``edge_w.get((a, b), edge_w.get((b, a), 0.0))`` tries both
    # orderings but some callers rely on symmetry).
    mst_sym = mst + mst.T
    return mst, mst_sym.tocsr()


def _project_point_to_line_segment(p, A, B):
    """Project point p onto line segment [A, B]."""
    AB = B - A
    AB_sq = np.dot(AB, AB)
    if AB_sq == 0:
        return A.copy()
    Ap = p - A
    t = np.dot(Ap, AB) / AB_sq
    t = np.clip(t, 0, 1)
    return A + t * AB


def _proj_point_on_line(p, A, B):
    """Project point p onto infinite line through A and B."""
    AB = B - A
    AB_sq = np.dot(AB, AB)
    if AB_sq == 0:
        return A.copy()
    Ap = p - A
    t = np.dot(Ap, AB) / AB_sq
    return A + t * AB


def _project_cells_to_mst(adata):
    """
    Project each cell onto the nearest edge of the MST (like project2MST in R).
    Updates adata with projected coordinates and new MST.
    """
    monocle = adata.uns['monocle']
    Z = monocle['reducedDimS']  # (dim, N)
    Y = monocle['reducedDimK']  # (dim, K)
    mst = monocle['mst']
    closest_vertex = monocle['pr_graph_cell_proj_closest_vertex']

    N = Z.shape[1]
    dim = Z.shape[0]
    K = Y.shape[1]

    # Get MST edge list
    tip_leaves = [v.index for v in mst.vs if mst.degree(v) == 1]

    P = np.zeros((dim, N))

    for i in range(N):
        cv = closest_vertex[i]
        cv_name = mst.vs[cv]['name']
        neighbors = mst.neighbors(cv)

        best_proj = None
        best_dist = np.inf

        Z_i = Z[:, i]

        for neighbor in neighbors:
            A = Y[:, cv]
            B = Y[:, neighbor]

            if cv in tip_leaves:
                proj = _proj_point_on_line(Z_i, A, B)
            else:
                proj = _project_point_to_line_segment(Z_i, A, B)

            d = np.linalg.norm(Z_i - proj)
            if d < best_dist:
                best_dist = d
                best_proj = proj

        if best_proj is None:
            P[:, i] = Y[:, cv]
        else:
            P[:, i] = best_proj

    # Build MST over projected cells.
    #
    # Mathematical fact: for any dimension,
    #     Euclidean MST ⊆ Relative-Neighbour Graph
    #                    ⊆ Gabriel Graph
    #                    ⊆ Delaunay Triangulation.
    # So every MST edge is a Delaunay edge. Running scipy's MST on the
    # Delaunay edge set gives an output identical (up to float
    # round-off) to running it on the full N×N pairwise-distance matrix,
    # at O(N·d) memory instead of O(N^2).
    #
    # R Monocle 2's project2MST shifts every distance by the smallest
    # positive pairwise distance to avoid zero-weight edges:
    #     dp <- dp + min(dp[dp != 0]); diag(dp) <- 0
    # The shift is a constant on every edge, so MST topology is
    # unchanged, but MST edge weights increase by min_dist. Pseudotime
    # (the cumulative MST-edge length from root) is therefore sensitive
    # to this shift, so we apply the same offset for bitwise agreement
    # with R's pseudotime values.
    N_cells = P.shape[1]
    pts = P.T                                           # (N, d)
    mst_sp, projected_dp_sparse = _euclidean_mst_delaunay(pts, N_cells)

    cell_mst = ig.Graph(n=N_cells, directed=False)
    cell_mst.vs['name'] = list(adata.obs_names)
    cell_mst.add_edges(list(zip(mst_sp.row.tolist(), mst_sp.col.tolist())))
    cell_mst.es['weight'] = mst_sp.data.tolist()

    # Store the MST-only sparse distance matrix — ``_extract_ddrtree_ordering``
    # walks MST edges and looks up distances here. It already accepts
    # either a dense ndarray or a scipy sparse matrix.
    monocle['pr_graph_cell_proj_tree'] = cell_mst
    monocle['pr_graph_cell_proj_dist'] = P
    monocle['projected_dp'] = projected_dp_sparse   # sparse N×N, MST-only

    return adata


def _extract_ddrtree_ordering(adata, root_cell_name, use_cell_mst=False):
    """
    Extract pseudotime ordering from the MST via BFS traversal from the root.

    Parameters
    ----------
    adata : AnnData
    root_cell_name : str
        Name of the root vertex in the MST.
    use_cell_mst : bool
        If True, use the cell projection MST instead of the Y-center MST.

    Returns
    -------
    pd.DataFrame with columns: sample_name, cell_state, pseudo_time, parent
    """
    monocle = adata.uns['monocle']

    if use_cell_mst:
        mst = monocle['pr_graph_cell_proj_tree']
        dp = monocle['projected_dp']
    else:
        mst = monocle['mst']
        dp = monocle['cellPairwiseDistances']

    n_vertices = mst.vcount()
    vertex_names = mst.vs['name']

    # Find root vertex index
    root_idx = vertex_names.index(root_cell_name)

    # BFS traversal: returns (order, layers, parents)
    bfs_result = mst.bfs(root_idx)
    order = bfs_result[0]
    fathers = bfs_result[2]   # parent of each vertex in the BFS tree

    pseudotimes = np.zeros(n_vertices)
    states = np.zeros(n_vertices, dtype=int)
    parents = [None] * n_vertices

    # --- State assignment matches R Monocle2 ---------------------------
    # R: a new state is entered when crossing an edge OUT of a branch
    # point (deg>2). All cells downstream of a branch point on the same
    # outgoing edge share a state. Every child of a branch point gets
    # its own fresh state id — including when the *root* itself is a
    # branch point (deg ≥ 3), in which case each subtree starts in a
    # different state rather than sharing the root's state.
    state_of = {}                     # node_idx -> state id
    next_state = 1
    root_is_branch = mst.degree(root_idx) >= 3
    if not root_is_branch:
        # Linear or single-child root → root seeds state 1
        state_of[root_idx] = next_state
        next_state += 1
    # Otherwise the root is only given a state once we step to a child.
    # state_of[root_idx] stays unset so subsequent lookups must go
    # through the propagation rule below.

    for node_idx in order:
        if node_idx < 0:
            continue
        if node_idx == root_idx:
            if node_idx in state_of:
                continue
            # Degenerate: root only (empty tree)
            state_of[node_idx] = 1
            continue
        father_idx = fathers[node_idx]
        if father_idx < 0:
            # Disconnected; inherit or seed state 1
            state_of[node_idx] = state_of.get(root_idx, 1)
            continue
        # If the *parent* is a branch point, this edge starts a new state.
        # This now applies even when the parent IS the root — matches R.
        if mst.degree(father_idx) >= 3:
            state_of[node_idx] = next_state
            next_state += 1
        else:
            # Non-branch parent → inherit the parent's state.
            # (If root was a branch point and is the parent here, the
            # `>= 3` branch above minted a fresh state; otherwise root
            # was seeded with state 1 and we inherit it here.)
            state_of[node_idx] = state_of.get(father_idx, 1)

    # If the root never got a state (branch-point root with no direct
    # state assignment), give it state 1 for completeness
    state_of.setdefault(root_idx, 1)

    # dp may be dense (cellPairwiseDistances on Y-centres) or sparse
    # (projected_dp on cells). Build a fast edge-weight lookup that
    # works for both: the MST has N-1 edges, walk them once.
    from scipy import sparse as _sp
    if _sp.issparse(dp):
        dp_coo = dp.tocoo()
        edge_w = {}
        for i, j, v in zip(dp_coo.row, dp_coo.col, dp_coo.data):
            edge_w[(int(i), int(j))] = float(v)
        def _edge(a, b):
            return edge_w.get((a, b), edge_w.get((b, a), 0.0))
    else:
        def _edge(a, b):
            if dp.shape[0] > max(a, b):
                return float(dp[a, b])
            return 0.0

    # Pseudotime: cumulative edge length from root
    for node_idx in order:
        if node_idx < 0:
            continue
        node_name = vertex_names[node_idx]
        father_idx = fathers[node_idx]

        if father_idx >= 0:
            parent_name = vertex_names[father_idx]
            parent_pseudotime = pseudotimes[father_idx]
            d = _edge(node_idx, father_idx)
            curr_node_pseudotime = parent_pseudotime + d
        else:
            parent_name = None
            curr_node_pseudotime = 0

        pseudotimes[node_idx] = curr_node_pseudotime
        states[node_idx] = state_of.get(node_idx, 1)
        parents[node_idx] = parent_name

    ordering_df = pd.DataFrame({
        'sample_name': vertex_names,
        'cell_state': states,
        'pseudo_time': pseudotimes,
        'parent': parents,
    })
    ordering_df.index = vertex_names
    return ordering_df


def _select_root_cell(adata, root_state=None, reverse=False, root_cell=None):
    """Select root cell for pseudotime ordering."""
    monocle = adata.uns['monocle']

    if root_cell is not None:
        root_cell = str(root_cell)
        if root_cell not in adata.obs_names:
            raise ValueError(f"Unknown root_cell {root_cell!r}")
        if monocle['dim_reduce_type'] == 'DDRTree':
            closest = np.asarray(
                monocle['pr_graph_cell_proj_closest_vertex']
            ).ravel().astype(int)
            cell_idx = int(adata.obs_names.get_loc(root_cell))
            return monocle['mst'].vs[int(closest[cell_idx])]['name']
        return root_cell
    elif root_state is not None:
        if 'State' not in adata.obs.columns:
            raise ValueError("State not set. Call order_cells() without root_state first.")

        # State is stored as a string-keyed Categorical (see writer in
        # this module); accept either int or str from the caller by
        # stringifying on comparison.
        root_state_key = str(root_state)
        root_candidates = adata.obs[adata.obs['State'].astype(str)
                                    == root_state_key]
        if len(root_candidates) == 0:
            raise ValueError(f"No cells for State = {root_state}")

        mst = monocle['mst']
        tip_leaves = [mst.vs[v]['name'] for v in range(mst.vcount()) if mst.degree(v) == 1]

        if 'Pseudotime' in adata.obs.columns:
            root_cell = root_candidates['Pseudotime'].idxmin()
        else:
            root_cell = root_candidates.index[0]

        if monocle['dim_reduce_type'] == 'DDRTree':
            closest = monocle['pr_graph_cell_proj_closest_vertex']
            cell_idx = list(adata.obs_names).index(root_cell)
            graph_point = closest[cell_idx]
            root_cell = mst.vs[graph_point]['name']

        return root_cell
    else:
        mst = monocle['mst']
        # Match R Monocle 2's behaviour exactly: default root = the
        # first vertex of the diameter path, `reverse=True` flips to
        # the last vertex. R's `select_root_cell`:
        #
        #   diameter <- get.diameter(minSpanningTree(cds))
        #   root_cell <- if (reverse) names(diameter[length(diameter)])
        #                else names(diameter[1])
        #
        # If the resulting pseudotime direction disagrees with a known
        # experimental variable (e.g. Hours), call
        # ``order_cells(reverse=True)`` or pass ``root_state=N``.
        diameter_path = mst.get_diameter(directed=False)
        if len(diameter_path) == 0:
            return mst.vs[0]['name']
        if reverse is True:
            return mst.vs[diameter_path[-1]]['name']
        return mst.vs[diameter_path[0]]['name']


def order_cells(adata, root_state=None, reverse=None, root_cell=None):
    """
    Order cells along the learned trajectory and assign pseudotime.

    Parameters
    ----------
    adata : AnnData
    root_state : int or None
        If specified, use cells in this state as root.
    root_cell : str or None
        If specified, anchor pseudotime at this exact observation. This is
        used by ``root_by_column`` so a mixed State cannot silently choose a
        root from the wrong metadata population.
    reverse : bool or None
        If True, reverse the ordering direction.

    Returns
    -------
    adata with Pseudotime and State in .obs
    """
    monocle = adata.uns['monocle']
    dim_reduce_type = monocle.get('dim_reduce_type')

    if dim_reduce_type is None:
        raise ValueError("Dimensionality not yet reduced. Call reduce_dimension() first.")

    if dim_reduce_type == 'DDRTree':
        Z = monocle['reducedDimS']
        Y = monocle['reducedDimK']
        K = Y.shape[1]
        mst = monocle['mst']

        # Bounds-check closest_vertex BEFORE using it. A corrupted value
        # (>= K) would otherwise surface deep inside igraph or numpy as
        # an opaque IndexError. Catch it here with a clear message.
        _cv = monocle.get('pr_graph_cell_proj_closest_vertex')
        if _cv is not None:
            _cv_arr = np.asarray(_cv).ravel().astype(int)
            if _cv_arr.size and (_cv_arr.min() < 0 or _cv_arr.max() >= K):
                raise AssertionError(
                    f"pr_graph_cell_proj_closest_vertex out of range: "
                    f"min={_cv_arr.min()}, max={_cv_arr.max()}, but the "
                    f"Y-centre MST has {K} vertices."
                )

        # Initial ordering on Y-center MST
        root_center = _select_root_cell(
            adata, root_state, reverse, root_cell=root_cell
        )
        cc_ordering = _extract_ddrtree_ordering(
            adata, root_center, use_cell_mst=False
        )

        # Project cells onto MST edges
        _project_cells_to_mst(adata)

        cell_mst = monocle['pr_graph_cell_proj_tree']
        closest_vertex = monocle['pr_graph_cell_proj_closest_vertex']

        # Find root in cell MST
        old_root_idx = list(mst.vs['name']).index(root_center)
        cells_at_root = np.where(closest_vertex == old_root_idx)[0]

        tip_leaves_cell = [v.index for v in cell_mst.vs if cell_mst.degree(v) == 1]
        tip_names = [cell_mst.vs[v]['name'] for v in tip_leaves_cell]

        root_cell_new = str(root_cell) if root_cell is not None else None
        # Preferred: a cell mapped to the root Y-centre that is also a
        # tip of the cell MST (matches R Monocle 2)
        for cell_idx in cells_at_root:
            cell_name = adata.obs_names[cell_idx]
            if cell_name in tip_names:
                root_cell_new = cell_name
                break
        # Fallback: any cell at the root Y-centre
        if root_cell_new is None and len(cells_at_root) > 0:
            root_cell_new = adata.obs_names[int(cells_at_root[0])]
        # Final fallback: use the cell closest (in pseudotime) to
        # the Y-root among tip cells
        if root_cell_new is None:
            if tip_names:
                root_cell_new = tip_names[0]
            else:
                root_cell_new = cell_mst.vs[0]['name']

        monocle['root_cell'] = root_cell_new

        # Extract ordering on cell projection MST
        cc_ordering_new = _extract_ddrtree_ordering(adata, root_cell_new, use_cell_mst=True)

        # Assign pseudotime
        adata.obs['Pseudotime'] = cc_ordering_new.loc[adata.obs_names, 'pseudo_time'].values

        # Assign state based on Y-center MST structure.
        # cc_ordering is keyed by Y-centre *vertex name* ("Y_0", ..., "Y_K-1"),
        # not by positional index. closest_vertex is an array of integer
        # Y-centre indices. Bounds-check, then look up by name.
        if root_state is None and root_cell is None:
            n_Y = mst.vcount()
            cv = np.asarray(closest_vertex).ravel().astype(int)
            if cv.size and (cv.min() < 0 or cv.max() >= n_Y):
                raise AssertionError(
                    f"closest_vertex out of range: min={cv.min()}, "
                    f"max={cv.max()}, but MST has {n_Y} vertices"
                )
            y_names = [mst.vs[i]['name'] for i in cv]
            cell_states = cc_ordering.loc[y_names, 'cell_state'].values
            # Store State as a **string-keyed** Categorical. Integer
            # categories leak into `adata.uns['State_colors']` lookups
            # downstream — ov.pl.embedding's palette normaliser stringifies
            # category keys but not the loop-over-categories in the
            # legend path, so int-keyed Categoricals crash with
            # `KeyError: <int>`. Keep State as the R-style "1","2",...
            adata.obs['State'] = pd.Categorical(
                [str(v) for v in cell_states]
            )
        else:
            adata.obs['State'] = pd.Categorical(
                [str(v) for v in
                 cc_ordering_new.loc[adata.obs_names, 'cell_state'].values]
            )

        # Find branch points. A strictly linear trajectory has no branch
        # points — warn the user instead of letting downstream code
        # silently index into an empty list.
        branch_points = [mst.vs[v]['name']
                         for v in range(mst.vcount())
                         if mst.degree(v) > 2]
        if not branch_points:
            import warnings as _warnings
            _warnings.warn(
                "No branch points detected; trajectory appears linear.",
                UserWarning,
            )
        monocle['branch_points'] = branch_points

    elif dim_reduce_type == 'ICA':
        mst = monocle['mst']
        selected_root_cell = _select_root_cell(
            adata, root_state, reverse, root_cell=root_cell
        )
        monocle['root_cell'] = selected_root_cell

        dp = monocle['cellPairwiseDistances']
        cc_ordering = _extract_ddrtree_ordering(
            adata, selected_root_cell, use_cell_mst=False
        )

        adata.obs['Pseudotime'] = cc_ordering.loc[adata.obs_names, 'pseudo_time'].values
        adata.obs['State'] = pd.Categorical(
            [str(v) for v in
             cc_ordering.loc[adata.obs_names, 'cell_state'].values]
        )

        branch_points = [mst.vs[v]['name'] for v in range(mst.vcount()) if mst.degree(v) > 2]
        monocle['branch_points'] = branch_points

    return adata
