r"""Tree layout for Bonsai results.

Two routes to 2-D coordinates:

``scout_layout`` — upstream's own. It runs Bonsai-scout's equal-angle pass
followed by **equal-daylight** refinement, then maps the result into the
**Poincaré disk** with Bonsai's own transform. That is the arrangement the
Bonsai-scout app shows: branches fan out, leaves crowd towards the rim, and the
whole tree is bounded by a circle. Preferred, and the default.

``equal_angle_layout`` — a small self-contained fallback in Euclidean space, for
when the scout module is unavailable. Equal-angle alone leaves wide empty wedges
next to crowded ones, which is exactly what the daylight pass fixes.

Drawing lives in :func:`omicverse.pl.bonsai`.
"""
from __future__ import annotations

import numpy as np

__all__ = ["equal_angle_layout", "scout_layout"]


def _adjacency(edges, lengths, n_vert):
    adj = [[] for _ in range(n_vert)]
    for (a, b), L in zip(edges, lengths):
        adj[a].append((b, float(L)))
        adj[b].append((a, float(L)))
    return adj


def _rooted(edges, lengths, n_vert, root=None):
    """Return ``(adj, parent, order, degree, root)`` for a tree rooted anywhere.

    The traversal is iterative: a Bonsai tree on a real dataset is deep enough
    that recursion hits the interpreter's stack limit.
    """
    adj = _adjacency(edges, lengths, n_vert)
    degree = np.array([len(a) for a in adj])
    if root is None:
        # An internal vertex centres the drawing better than a leaf, and the
        # highest-degree one keeps the initial wedges wide.
        root = int(np.argmax(degree))

    parent = np.full(n_vert, -1, dtype=int)
    t_parent = np.zeros(n_vert, dtype=float)
    order, stack, seen = [], [root], np.zeros(n_vert, dtype=bool)
    seen[root] = True
    while stack:
        v = stack.pop()
        order.append(v)
        for w, L in adj[v]:
            if not seen[w]:
                seen[w] = True
                parent[w] = v
                t_parent[w] = L
                stack.append(w)
    return adj, parent, t_parent, order, degree, root


def equal_angle_layout(edges, lengths, n_vert, root=None):
    r"""Lay an unrooted tree out in 2-D, preserving branch lengths.

    Each subtree receives an angular wedge sized by its share of the leaves, and
    each vertex sits one branch length from its parent along that wedge's
    bisector. Distance along the drawing is therefore distance in the model.

    Returns
    -------
    tuple
        ``(pos, parent, degree, root)`` — ``pos`` is ``(n_vert, 2)``.
    """
    adj, parent, _, order, degree, root = _rooted(edges, lengths, n_vert, root)

    n_leaves = np.zeros(n_vert, dtype=float)
    for v in reversed(order):                       # children before parents
        if degree[v] == 1 and v != root:
            n_leaves[v] = 1.0
        if parent[v] >= 0:
            n_leaves[parent[v]] += max(n_leaves[v], 1.0 if degree[v] == 1 else 0.0)
    n_leaves[n_leaves == 0] = 1.0

    pos = np.zeros((n_vert, 2), dtype=float)
    lo = np.zeros(n_vert)
    hi = np.zeros(n_vert)
    hi[root] = 2.0 * np.pi
    for v in order:
        kids = [(w, L) for w, L in adj[v] if parent[w] == v]
        if not kids:
            continue
        total = sum(max(n_leaves[w], 1.0) for w, _ in kids)
        a = lo[v]
        for w, L in kids:
            share = (hi[v] - lo[v]) * max(n_leaves[w], 1.0) / total
            lo[w], hi[w] = a, a + share
            mid = a + share / 2.0
            pos[w] = pos[v] + L * np.array([np.cos(mid), np.sin(mid)])
            a += share
    return pos, parent, degree, root


def _build_layout_tree(edges, lengths, n_vert, root=None):
    """Assemble Bonsai-scout's ``Layout_Tree`` from a bare edge list.

    Upstream builds it from an ``SCData`` tree, which we do not have after the
    run: what survives in ``adata.uns`` is ``edgeInfo.txt``. The node fields the
    layout actually reads are the branch length to the parent (``tParent``), the
    child list and the leaf/root flags, so the tree can be rebuilt from edges
    alone.
    """
    from .scout.my_tree_layout import Layout_Tree, Layout_TreeNode

    adj, parent, t_parent, order, degree, root = _rooted(edges, lengths, n_vert, root)

    ly = Layout_Tree()
    ly.nNodes = int(n_vert)
    nodes = {}
    ly.root = nodes[root] = Layout_TreeNode(
        vert_ind=int(root), childNodes=[], parentNode=None,
        isLeaf=False, isRoot=True, tParent=None, nodeId=int(root), opt_angle=360)
    for v in order:                                  # parents before children
        if v == root:
            continue
        p = nodes[int(parent[v])]
        node = Layout_TreeNode(
            vert_ind=int(v), childNodes=[], parentNode=p,
            isLeaf=bool(degree[v] == 1), isRoot=False,
            tParent=float(t_parent[v]), nodeId=int(v))
        p.childNodes.append(node)
        nodes[int(v)] = node
    ly.root.rearrange_branches_node(flipped_node_ids=[], ladderize_all=True,
                                    nNodes=ly.nNodes)
    return ly


def scout_layout(edges, lengths, n_vert, root=None, *, equal_daylight=True,
                 poincare=True, frac_within=0.8, within_radius=0.8,
                 max_steps=100, verbose=False):
    r"""Lay the tree out the way Bonsai-scout does.

    Equal-angle, then equal-daylight refinement, then the Poincaré-disk mapping —
    all three are upstream's own code, vendored under ``scout/`` and
    ``bonsai/bonsai_helpers.py``.

    The daylight pass is what makes the drawing readable: equal-angle alone
    leaves one subtree crammed against another while a neighbouring wedge sits
    empty, and daylight equalises the gaps a vertex sees between its subtrees.
    The Poincaré step then bounds the tree in a unit circle, which is why deep
    branches compress towards the rim instead of running off the axes.

    Parameters
    ----------
    edges, lengths, n_vert
        Edge list, branch lengths and vertex count, as stored by
        :func:`omicverse.tl.bonsai`.
    root
        Vertex to centre on; defaults to the highest-degree vertex.
    equal_daylight
        Run the daylight refinement. Off gives the plain equal-angle layout.
    poincare
        Map into the Poincaré disk. Off leaves Euclidean coordinates.
    frac_within, within_radius
        The Poincaré zoom is chosen so ``frac_within`` of the vertices land
        inside ``within_radius``. Upstream's app defaults to ``0.8``/``0.8``,
        matching upstream's app. Raise them to push the tree towards the rim,
        lower them to pull it in.
    max_steps
        Daylight iterations; upstream's own default is 100.
    verbose
        Pass upstream's verbosity through.

    Returns
    -------
    tuple
        ``(pos, parent, degree, root)`` — ``pos`` is ``(n_vert, 2)``, inside the
        unit disk when ``poincare`` is true.
    """
    import logging

    _, parent, _, _, degree, root = _rooted(edges, lengths, n_vert, root)

    ly = _build_layout_tree(edges, lengths, n_vert, root)
    # Upstream logs every daylight iteration at DEBUG through its own module
    # logger, which ignores the verbose flag; quiet it for the duration.
    up_log = logging.getLogger("myapp")   # upstream names its logger this
    prev = up_log.level
    if not verbose:
        up_log.setLevel(logging.WARNING)
    try:
        ly.equalAngle(get_nodelist=equal_daylight, verbose=verbose)
        if equal_daylight:
            ly = ly.equalDaylightAll(verbose=verbose, max_steps=max_steps)
    finally:
        up_log.setLevel(prev)
    pos = np.asarray(ly.coords, dtype=float)

    if poincare:
        import contextlib
        import io

        from .bonsai.bonsai_helpers import (get_centroid_poincare,
                                            get_radial_poincare,
                                            transform_coords_poincare)
        # Upstream solves for zoom and origin together, but that search is
        # unreliable on a tree whose subtrees differ a lot in depth: it settles
        # on an origin far from the layout's centroid and the drawing comes out
        # lopsided. Splitting the two is steadier -- the centroid is the origin,
        # and the zoom is then just the scalar that puts `frac_within` of the
        # vertices inside `within_radius`. Both primitives are upstream's; only
        # the outer loop is ours. It also keeps upstream's stdout notes quiet.
        sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with sink:
            origin = np.asarray(get_centroid_poincare(pos, zoom=1.0), dtype=float)
            q = float(np.clip(frac_within, 0.0, 1.0))

            def _radius_at(z):
                return float(np.quantile(
                    get_radial_poincare(pos, origin=origin, zoom=z), q))

            lo_z, hi_z = 1e-3, 1e3          # radius grows monotonically with zoom
            for _ in range(60):
                mid = np.sqrt(lo_z * hi_z)
                if _radius_at(mid) < within_radius:
                    lo_z = mid
                else:
                    hi_z = mid
            pos = transform_coords_poincare(pos, origin=origin,
                                            zoom=np.sqrt(lo_z * hi_z))
    return pos, parent, degree, root
