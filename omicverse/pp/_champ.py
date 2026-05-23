"""Convex Hull of Admissible Modularity Partitions — ``ov.pp.champ``.

Implementation of Weir, Emmons, Wakefield, Hopkins & Mucha (2017),
"Post-processing partitions to identify domains of modularity
optimization" (*Algorithms* 10(3):93). The key observation is purely
geometric: for any **fixed** partition :math:`P`, Newman modularity

.. math::

   Q(\\gamma; P) =
       \\frac{1}{2m} \\sum_{ij}\\bigl[A_{ij} - \\gamma\\,
                                    \\frac{d_i d_j}{2m}\\bigr]
       \\delta(c_i, c_j)
       \\;=\\; a_P \\;-\\; \\gamma\\, b_P

is a **linear** function of the resolution parameter
:math:`\\gamma`. A family of candidate partitions therefore corresponds
to a family of lines in the :math:`(\\gamma, Q)` plane; the partition
that is modularity-optimal at *any* given :math:`\\gamma` lies on the
**upper envelope** of those lines, which is dual to the **upper convex
hull** of the points :math:`(b_P, a_P)`.

CHAMP runs leiden at many candidate :math:`\\gamma` values, computes
:math:`(a_P, b_P)` for each resulting partition, finds the upper hull,
and returns the hull partition whose **admissible γ-range** is widest
— the partition that's modularity-optimal across the broadest band of
resolutions.

The two complementary resolution-selection paths in omicverse:

- ``ov.single.auto_resolution`` — *stochastic* / *bootstrap-stability*:
  measures how reproducible each resolution's clustering is under
  data perturbation, with a null-model adjustment per Lange et al.
  2004. Picks the **most reproducible** resolution.

- ``ov.pp.champ`` — *deterministic* / *modularity-geometric*: no Monte
  Carlo, no null permutation. Picks the **most modularity-stable**
  partition by analysing the convex structure of the candidate set.

They answer different questions and can disagree; either is defensible.
"""
from __future__ import annotations

import contextlib
import io
from typing import Optional, Sequence, Tuple

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .._registry import register_function
from .._settings import EMOJI, add_reference
from ..report._provenance import tracked, note


# ────────────────────── modularity coefficients ───────────────────────────────


def _modularity_coefficients(W, labels, modularity: str = 'newman'
                              ) -> tuple[float, float]:
    """Return ``(a, b)`` such that the chosen modularity is
    :math:`Q(\\gamma; P) = a - \\gamma\\,b`.

    Two flavours, both linear in γ:

    - ``'newman'`` — Newman-Girvan modularity:

      .. math::

          Q = \\frac{1}{2m}\\sum_{ij}\\bigl[A_{ij}
                  - \\gamma\\,\\frac{d_i d_j}{2m}\\bigr]\\delta(c_i, c_j)

      with ``a = within-cluster edge weight / 2m`` and
      ``b = Σ_c (Σ_{i in c} d_i)² / (2m)²``.

    - ``'cpm'`` — Constant Potts Model (Reichardt-Bornholdt 2006), which
      is **resolution-limit-free** (Fortunato & Barthelemy 2007):

      .. math::

          Q = \\sum_{ij}\\bigl[A_{ij} - \\gamma\\bigr]\\,\\delta(c_i, c_j)

      with ``a = within-cluster edge weight / W_total`` and
      ``b = Σ_c |c|² / N²``. Same normalisation pattern as Newman so
      γ is on a comparable order of magnitude.

    Linear in ``nnz(W)`` regardless of flavour: a single COO sweep
    plus a scatter-add over labels.
    """
    if not sp.issparse(W):
        W = sp.csr_matrix(W)
    coo = W.tocoo()
    total = float(W.sum())  # 2m for undirected
    if total <= 0:
        return 0.0, 0.0

    # a = within-cluster edge weight / total (same for both flavours).
    same = labels[coo.row] == labels[coo.col]
    a = float(coo.data[same].sum()) / total

    # b depends on the modularity flavour.
    unique_labels, inv = np.unique(labels, return_inverse=True)
    if modularity == 'newman':
        d = np.asarray(W.sum(axis=1)).ravel()
        D = np.bincount(inv, weights=d, minlength=len(unique_labels))
        b = float(np.sum(D * D)) / (total * total)
    elif modularity == 'cpm':
        # b_CPM = Σ_c |c|² / N²
        n = float(W.shape[0])
        sizes = np.bincount(inv, minlength=len(unique_labels))
        b = float(np.sum(sizes * sizes)) / (n * n)
    else:
        raise ValueError(
            f"modularity must be 'newman' (default) or 'cpm'; "
            f"got {modularity!r}."
        )
    return a, b


# ────────────────────── upper convex hull ──────────────────────────────────────


def _upper_hull_indices(b: np.ndarray, a: np.ndarray) -> list[int]:
    """Andrew's monotone-chain upper-hull pass.

    Returns the indices of points lying on the **upper** convex hull of
    the 2-D points ``(b_i, a_i)``, in order of ascending ``b``.
    Geometrically dual to the upper envelope of the lines
    :math:`y = a_i - b_i x`.
    """
    n = len(b)
    if n <= 2:
        return list(range(n))
    # Sort by b ascending; ties broken by a descending so the very
    # first point at each b-coordinate is the highest.
    order = sorted(range(n), key=lambda i: (b[i], -a[i]))
    hull: list[int] = []
    for i in order:
        while len(hull) >= 2:
            i1, i2 = hull[-2], hull[-1]
            # Cross product of (p2 - p1) × (p3 - p1).
            cross = ((b[i2] - b[i1]) * (a[i] - a[i1]) -
                      (a[i2] - a[i1]) * (b[i] - b[i1]))
            # Upper hull keeps RIGHT turns (cross < 0); pop otherwise.
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


# ───────────────────────────── public entry ───────────────────────────────────


@register_function(
    aliases=['CHAMP', 'champ', 'modularity convex hull',
             'admissible partitions'],
    category="preprocessing",
    description=(
        "Convex Hull of Admissible Modularity Partitions (Weir et al. "
        "2017). Generates candidate Leiden partitions across a γ grid, "
        "computes the (a, b) coefficients of Q(γ) = a − γ·b for each, "
        "finds their upper convex hull (the partitions modularity-"
        "optimal at *some* γ), and returns the hull partition with the "
        "widest admissible γ-range — the deterministic modularity-"
        "geometric counterpart to ov.single.auto_resolution's "
        "stochastic null-adjusted bootstrap stability."
    ),
    prerequisites={'functions': ['pp.neighbors']},
    requires={'obsp': ['connectivities']},
    produces={'obs': ['champ'], 'uns': ['champ']},
    auto_fix='auto',
    examples=[
        'partition, gamma_range, df = ov.pp.champ(adata)',
        'ov.pp.champ(adata, resolutions=np.linspace(0.05, 3.0, 30))',
    ],
    related=['pp.leiden', 'single.auto_resolution'],
)
@tracked('champ', 'ov.pp.champ')
def champ(
    adata: anndata.AnnData,
    resolutions: Optional[Sequence[float]] = None,
    *,
    n_partitions: int = 30,
    gamma_min: float = 0.05,
    gamma_max: float = 3.0,
    n_seeds: int = 1,
    modularity: str = 'newman',
    width_metric: str = 'log',
    adaptive: bool = False,
    adaptive_max_iter: int = 3,
    adaptive_n_refine: int = 3,
    key_added: str = 'champ',
    random_state: int = 0,
    verbose: bool = True,
) -> Tuple[anndata.AnnData, Tuple[float, float], pd.DataFrame]:
    r"""Pick the modularity-stablest Leiden partition via CHAMP
    (Weir et al. 2017).

    Algorithm
    ---------
    1. Run Leiden at ``n_partitions`` candidate γ values
       evenly spaced over :math:`[\gamma_\min,\,\gamma_\max]` (or use
       the explicit grid passed in ``resolutions``). Deduplicate
       partitions that produce identical labels.
    2. For each unique partition :math:`P`, compute
       :math:`(a_P, b_P)` such that

       .. math::

           Q(\gamma; P) = a_P - \gamma\, b_P.

    3. Compute the **upper convex hull** of the points
       :math:`\{(b_P, a_P)\}`. The hull's vertices are the partitions
       that are modularity-optimal for *some* γ; everything else is
       dominated.
    4. For each consecutive pair of hull vertices, compute the γ at
       which their lines intersect:

       .. math::

           \gamma_{i,i+1} = \frac{a_{i+1} - a_i}{b_{i+1} - b_i}.

    5. Each hull partition's **admissible γ-range** is the interval
       between its two adjacent crossover γ's (capped at
       :math:`[0,\,\gamma_\max]`).
    6. Return the hull partition with the **widest admissible range**.

    No Monte Carlo, no null model — purely a geometric statement about
    the modularity landscape's convex structure.

    Parameters
    ----------
    adata
        AnnData with a precomputed neighbor graph
        (``adata.obsp['connectivities']``).
    resolutions
        Explicit γ grid; overrides ``n_partitions`` /
        ``gamma_min`` / ``gamma_max``.
    n_partitions
        Number of γ values to scan when ``resolutions`` is ``None``.
    gamma_min, gamma_max
        Endpoints of the γ scan; ``gamma_max`` also caps the leftmost
        hull partition's "open" admissible range so widths are finite.
    n_seeds
        Number of random seeds to try at *each* γ. Leiden is heuristic
        and converges to a local modularity maximum; running multiple
        seeds at the same γ can surface different partitions and gives
        CHAMP a denser candidate cloud to find the true upper hull on.
        Default 1; the original Weir et al. 2017 paper recommends 3-5.
        Cost scales linearly: ``n_seeds × n_partitions`` leiden calls.
    modularity
        ``'newman'`` (default) — Newman-Girvan modularity. ``'cpm'`` —
        Constant Potts Model (Reichardt-Bornholdt 2006), which is
        **resolution-limit-free** in the sense of Fortunato &
        Barthelemy 2007. Both have the same linear-in-γ structure;
        ``'cpm'`` writes ``b_P = Σ_c |c|² / N²`` instead of degree
        squared, so γ-units differ — narrower γ ranges typically work.
        For ``'cpm'`` the candidate generator passes
        ``partition_type=leidenalg.CPMVertexPartition`` to leiden so
        that candidates and scoring share an objective.
    width_metric
        How "widest admissible γ-range" is measured when picking the
        chosen partition. Choices:

        - ``'log'`` (**default**, omicverse): multiplicative width
          :math:`\log(\gamma_{hi}/\gamma_{lo})`. Modularity is invariant
          under :math:`\gamma \mapsto c\gamma` (the optimal partition at
          each γ doesn't change), so γ has no natural additive scale —
          the canonical "width" on a scale-free axis is multiplicative.
          Matters in practice because additive width systematically
          over-rewards fine partitions: small differences in
          :math:`b` between fine partitions get amplified into large
          additive γ-ranges by the small denominator in the crossover
          formula :math:`\gamma = \Delta a / \Delta b`. The log metric
          undoes this geometric bias.
        - ``'linear'`` (Weir et al. 2017 canonical): additive width
          :math:`\gamma_{hi} - \gamma_{lo}`. Reproduces the original
          paper's behaviour. Tends to over-cluster on data with wide
          high-γ plateaus (typical for scRNA).
        - ``'relative'``: :math:`(\gamma_{hi}-\gamma_{lo}) /
          \overline{\gamma}` where :math:`\overline{\gamma}` is the
          midpoint. Closely related to log width; included for
          completeness.
    adaptive
        If ``True``, iteratively refine the γ-grid around the current
        hull's crossovers (active-set style): build the hull → for
        each crossover sample ``adaptive_n_refine`` extra γ
        values nearby → re-run leiden → recompute the hull → repeat
        until no new hull vertex appears or
        ``adaptive_max_iter`` iterations are reached. Catches
        hull partitions that uniform γ-sampling misses around sharp
        transitions.
    adaptive_max_iter
        Cap on the adaptive-refinement outer loop. Default 3.
    adaptive_n_refine
        Number of extra γ values sampled around each crossover per
        adaptive iteration. Default 3.
    key_added
        ``adata.obs`` column to write the chosen partition's labels to.
    random_state
        Seed for Leiden RNG.
    verbose
        Stream per-step progress.

    Returns
    -------
    Tuple[anndata.AnnData, Tuple[float, float], pandas.DataFrame]
        ``(adata, (gamma_lo, gamma_hi), partitions_df)`` where
        ``partitions_df`` is one row per unique candidate partition with
        columns ``a``, ``b``, ``n_clusters``, ``origin_resolution``,
        ``on_hull`` (bool), ``gamma_lo``, ``gamma_hi``,
        ``gamma_range``. Also writes ``adata.obs[key_added]`` and
        ``adata.uns['champ']``.

    References
    ----------
    Weir, Emmons, Wakefield, Hopkins, Mucha. "Post-processing partitions
    to identify domains of modularity optimization."
    *Algorithms* 10(3):93, 2017. https://doi.org/10.3390/a10030093
    """
    if 'connectivities' not in adata.obsp:
        raise ValueError(
            "champ requires a precomputed neighbor graph "
            "(adata.obsp['connectivities']); run ov.pp.neighbors first."
        )
    if not (0 <= gamma_min < gamma_max):
        raise ValueError(
            f"gamma_min/gamma_max must satisfy 0 <= min < max; got "
            f"({gamma_min}, {gamma_max})."
        )

    if resolutions is None:
        resolutions = list(np.linspace(gamma_min, gamma_max, n_partitions))
    resolutions = sorted(float(np.round(r, 4)) for r in resolutions)
    if len(resolutions) < 2:
        raise ValueError(
            "champ needs at least 2 distinct resolution values to "
            "compare partitions; got "
            f"{len(resolutions)} (after deduplication)."
        )

    if modularity not in ('newman', 'cpm'):
        raise ValueError(
            f"modularity must be 'newman' or 'cpm'; got {modularity!r}."
        )
    if n_seeds < 1:
        raise ValueError(f"n_seeds must be >= 1; got {n_seeds}.")
    if verbose:
        print(f"{EMOJI['start']} CHAMP: scanning {len(resolutions)} "
               f"resolutions × {n_seeds} seed(s)"
               f" ∈ [{gamma_min:.2f}, {gamma_max:.2f}]"
               f" ({modularity} modularity, {width_metric} width)")

    # ── Candidate-generation helper: appends NEW unique partitions
    # found at each (γ, seed) combo. Idempotent across calls (uses the
    # shared seen_signatures dict). Used both for the initial scan and
    # for adaptive refinement passes.
    from ..pp import leiden as _leiden  # tracked; nesting guard silences

    TMP_KEY = '_champ_tmp'
    _preexisting_tmp = (adata.obs[TMP_KEY].copy()
                         if TMP_KEY in adata.obs.columns else None)
    leiden_kwargs: dict = {}
    if modularity == 'cpm':
        try:
            import leidenalg
            leiden_kwargs['partition_type'] = leidenalg.CPMVertexPartition
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "modularity='cpm' requires `leidenalg` so the candidate "
                "partitions match the CPM scoring; install via "
                "`pip install leidenalg`."
            ) from exc

    def _generate(resolutions_to_try, partitions, seen_signatures):
        """Run leiden at each (γ, seed) combo, append unique partitions
        to ``partitions`` (in place)."""
        for r in resolutions_to_try:
            for s in range(n_seeds):
                with contextlib.redirect_stdout(io.StringIO()):
                    _leiden(adata, resolution=float(r),
                             key_added=TMP_KEY,
                             random_state=int(random_state + s),
                             **leiden_kwargs)
                labels = np.ascontiguousarray(
                    adata.obs[TMP_KEY].astype(np.int32).values
                )
                sig = labels.tobytes()
                if sig in seen_signatures:
                    continue
                seen_signatures[sig] = len(partitions)
                partitions.append({
                    'origin_resolution': float(r),
                    'origin_seed':       int(random_state + s),
                    'labels':            labels,
                    'n_clusters':        int(np.unique(labels).size),
                })

    partitions: list[dict] = []
    seen_signatures: dict[bytes, int] = {}
    _generate(resolutions, partitions, seen_signatures)
    if verbose:
        print(f"  initial: {len(partitions)} unique partitions "
               f"({len(resolutions) * n_seeds} leiden calls)")

    # 2. Compute (a, b) per modularity flavour.
    W = adata.obsp['connectivities']

    def _compute_ab(start: int = 0):
        for i in range(start, len(partitions)):
            p = partitions[i]
            a, b = _modularity_coefficients(W, p['labels'],
                                              modularity=modularity)
            p['a'] = a
            p['b'] = b

    _compute_ab()
    a_vals = np.array([p['a'] for p in partitions])
    b_vals = np.array([p['b'] for p in partitions])

    # 2b. Adaptive refinement (active-set on the hull).
    if adaptive:
        delta = (gamma_max - gamma_min) / max(2 * len(resolutions), 4)
        for it in range(adaptive_max_iter):
            hull_idx_iter = _upper_hull_indices(b_vals, a_vals)
            # Crossovers between consecutive hull partitions.
            cross = []
            for k in range(len(hull_idx_iter) - 1):
                i, j = hull_idx_iter[k], hull_idx_iter[k + 1]
                denom = b_vals[j] - b_vals[i]
                if denom != 0:
                    cross.append(float((a_vals[j] - a_vals[i]) / denom))
            # Sample γ values near each crossover.
            new_res = []
            for c in cross:
                if not (gamma_min <= c <= gamma_max):
                    continue
                for k in range(1, adaptive_n_refine + 1):
                    offset = delta * k / adaptive_n_refine
                    for g in (c - offset, c + offset):
                        if gamma_min <= g <= gamma_max:
                            new_res.append(round(g, 5))
            new_res = sorted(set(new_res))
            n_before = len(partitions)
            _generate(new_res, partitions, seen_signatures)
            if len(partitions) == n_before:
                if verbose:
                    print(f"  adaptive iter {it+1}: hull stable, stopping")
                break
            _compute_ab(n_before)
            a_vals = np.array([p['a'] for p in partitions])
            b_vals = np.array([p['b'] for p in partitions])
            if verbose:
                print(f"  adaptive iter {it+1}: +"
                       f"{len(partitions) - n_before} partitions")
            delta /= 2  # progressively tighter

    # Restore or drop the scratch column.
    if _preexisting_tmp is not None:
        adata.obs[TMP_KEY] = _preexisting_tmp
    elif TMP_KEY in adata.obs.columns:
        del adata.obs[TMP_KEY]

    if verbose:
        print(f"  total: {len(partitions)} unique partitions")

    # 3. Upper convex hull in (b, a) plane.
    hull_idx = _upper_hull_indices(b_vals, a_vals)
    H = len(hull_idx)
    if verbose:
        print(f"  {H} partitions on the upper convex hull (admissible)")

    # 4. Crossover γ between consecutive hull partitions.
    crossovers = []
    for k in range(H - 1):
        i, j = hull_idx[k], hull_idx[k + 1]
        denom = b_vals[j] - b_vals[i]
        crossovers.append(float((a_vals[j] - a_vals[i]) / denom)
                            if denom != 0 else float('inf'))

    # 5. Per-hull-partition admissible γ-range.
    on_hull = np.zeros(len(partitions), dtype=bool)
    on_hull[hull_idx] = True
    gamma_lo = np.full(len(partitions), np.nan)
    gamma_hi = np.full(len(partitions), np.nan)
    for k, hi_pos in enumerate(hull_idx):
        if H == 1:
            lo, hi = 0.0, gamma_max
        elif k == 0:
            # Leftmost (smallest b): admissible at γ > crossovers[0]
            lo = crossovers[0]
            hi = gamma_max
        elif k == H - 1:
            # Rightmost (largest b): admissible at γ < crossovers[-1]
            lo = 0.0
            hi = crossovers[-1]
        else:
            # Middle: bracketed by two crossovers
            lo = crossovers[k]
            hi = crossovers[k - 1]
        # Clamp to [0, gamma_max]. The upper clamp on gamma_lo matters
        # when two hull partitions share a 'b' value: their crossover is
        # +inf, which would otherwise propagate as NaN through log-widths.
        gamma_lo[hi_pos] = min(gamma_max, max(0.0, lo))
        gamma_hi[hi_pos] = max(0.0, min(gamma_max, hi))

    # 6. Pick the widest range under the chosen metric. Clamp negative
    # widths to zero (a hull partition whose admissible region extends
    # beyond [0, gamma_max] gets 0, not a confusing negative number)
    # and exclude non-hull rows from the argmax via -inf.
    if width_metric == 'linear':
        raw_widths = gamma_hi - gamma_lo
    elif width_metric == 'log':
        # γ-space is scale-free → multiplicative width. Clamp γ_lo at
        # gamma_min to avoid log(0) for the rightmost hull partition
        # (whose γ_lo we set to 0 by convention).
        lo_clamped = np.maximum(gamma_lo, gamma_min)
        hi_clamped = np.maximum(gamma_hi, lo_clamped)
        raw_widths = np.log(hi_clamped) - np.log(lo_clamped)
    elif width_metric == 'relative':
        midpoint = (gamma_hi + gamma_lo) / 2.0
        midpoint = np.maximum(midpoint, gamma_min)
        raw_widths = (gamma_hi - gamma_lo) / midpoint
    else:
        raise ValueError(
            f"width_metric must be 'log' (default), 'linear', or "
            f"'relative'; got {width_metric!r}."
        )
    widths = np.where(on_hull,
                       np.maximum(raw_widths, 0.0),
                       np.full(len(partitions), -np.inf))
    best_idx = int(np.argmax(widths))
    best_partition = partitions[best_idx]
    best_lo = float(gamma_lo[best_idx])
    best_hi = float(gamma_hi[best_idx])

    # ── Write back ──────────────────────────────────────────────────
    adata.obs[key_added] = pd.Categorical(
        best_partition['labels'].astype(str)
    )
    df = pd.DataFrame({
        'origin_resolution': [p['origin_resolution'] for p in partitions],
        'a':                  a_vals,
        'b':                  b_vals,
        'n_clusters':         [p['n_clusters'] for p in partitions],
        'on_hull':            on_hull,
        'gamma_lo':           gamma_lo,
        'gamma_hi':           gamma_hi,
        'gamma_range':        widths,
    }).sort_values('b').reset_index(drop=True)

    # Honour key_added for the uns slot too — scanpy convention. A
    # hard-coded 'champ' would clobber any previous run stored under a
    # different user-chosen key.
    adata.uns[key_added] = {
        'method': 'CHAMP (Weir et al. 2017)',
        'partitions':            df.to_dict('list'),
        'chosen_origin_resolution': float(best_partition['origin_resolution']),
        'chosen_n_clusters':     int(best_partition['n_clusters']),
        'chosen_gamma_range':    [best_lo, best_hi],
        'chosen_gamma_width':    best_hi - best_lo,
        'gamma_min':             float(gamma_min),
        'gamma_max':             float(gamma_max),
        'width_metric':          width_metric,
        'modularity':            modularity,
        'n_seeds':               int(n_seeds),
        'adaptive':              bool(adaptive),
    }
    add_reference(
        adata, key_added,
        f'CHAMP partition (γ-range [{best_lo:.3f}, {best_hi:.3f}], '
        f'{best_partition["n_clusters"]} clusters)',
    )

    if verbose:
        print(f"{EMOJI['done']} chosen partition: "
               f"{best_partition['n_clusters']} clusters; admissible "
               f"γ ∈ [{best_lo:.3f}, {best_hi:.3f}] "
               f"(width {best_hi - best_lo:.3f})")

    note(
        backend=f'CHAMP · {len(partitions)} partitions · '
                 f'{H} on hull',
        viz=[
            {'function': 'ov.pl.cluster_sizes_bar',
              'kwargs': {'groupby': key_added}},
            *([{'function': 'ov.pl.embedding',
                 'kwargs': {'basis': 'X_umap', 'color': key_added,
                            'frameon': 'small'}}]
               if 'X_umap' in adata.obsm else []),
        ],
    )

    return adata, (best_lo, best_hi), df
