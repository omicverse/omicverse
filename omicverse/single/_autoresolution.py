"""Pick the most reproducible Leiden resolution via null-adjusted
bootstrap-ARI stability — ``ov.single.autoResolution``.

Implementation follows Lange, Roth, Braun & Buhmann (*Neural Computation*
16(6):1299–1323, 2004), "Stability-Based Validation of Clustering
Solutions". The core observation is that bootstrap stability alone
systematically prefers fine partitions: small tight clusters are
mechanically reproducible under any subsampling procedure regardless
of whether they reflect biological structure. The Lange–Buhmann fix is
to subtract a procedurally-matched **null** stability — what's left is
the above-chance reproducibility of the real cluster structure, which
is what "auto resolution" should actually mean.

The historical scDrug-inherited implementation in ``_scdrug.py`` had
several issues — silhouette over a per-cell co-clustering distance
matrix (double-dipping, :math:`O(n^2)` memory), a ``best_resolution=0``
init bug when all silhouettes are negative, and ``mp.Pool`` overhead.
This module is a clean rewrite kept separate from the drug-response
code it shipped with.
"""
from __future__ import annotations

from typing import Optional, Sequence

import anndata
import numpy as np
import pandas as pd

from .._registry import register_function
from .._settings import EMOJI, add_reference
from ..report._provenance import tracked, note


# ───────────────────── helpers shared by autoResolution ───────────────────────


def _subsample_indices(n_obs: int, n_subsamples: int, subsample_frac: float,
                        rng: np.random.Generator) -> list[np.ndarray]:
    """Independent without-replacement subsamples of ``adata.obs`` indices."""
    sub_n = max(int(round(n_obs * subsample_frac)), 30)
    return [
        np.sort(rng.choice(n_obs, size=sub_n, replace=False))
        for _ in range(n_subsamples)
    ]


def _bootstrap_stability_pass(
    adata: anndata.AnnData,
    resolutions: Sequence[float],
    subsamples: Sequence[np.ndarray],
    random_state: int,
    label: str = '',
    verbose: bool = False,
):
    """Per-resolution bootstrap stability on the given AnnData.

    For each ``r``:

    - Run leiden on the full graph → reference labels at ``r``.
    - For each subsample index array, run leiden on the induced subgraph,
      score ARI vs the reference restricted to the subsample.
    - Stability(r) = mean ARI across subsamples.

    Returns
    -------
    scores : dict[float, dict]
        ``{r: {'mean_ari': float, 'std_ari': float, 'n_clusters': int}}``.
    refs   : dict[float, np.ndarray]
        Reference labels at each ``r`` (kept so the caller can write the
        winning resolution's labels back without re-clustering).
    """
    from sklearn.metrics import adjusted_rand_score
    from ..pp import leiden as _leiden  # tracked; nested calls are silenced

    REF_KEY = '_autores_ref'
    SUB_KEY = '_autores_sub'
    scores: dict[float, dict] = {}
    refs: dict[float, np.ndarray] = {}
    tag = f'[{label}] ' if label else ''

    for r in resolutions:
        _leiden(adata, resolution=r, key_added=REF_KEY,
                random_state=random_state)
        ref = adata.obs[REF_KEY].astype(str).values.copy()
        n_clusters = int(pd.Series(ref).nunique())
        refs[r] = ref

        aris = []
        for idx in subsamples:
            sub = adata[idx].copy()
            _leiden(sub, resolution=r, key_added=SUB_KEY,
                    random_state=random_state)
            aris.append(adjusted_rand_score(
                ref[idx],
                sub.obs[SUB_KEY].astype(str).values,
            ))

        mean_ari = float(np.mean(aris))
        std_ari = float(np.std(aris))
        scores[r] = {
            'mean_ari':   mean_ari,
            'std_ari':    std_ari,
            'n_clusters': n_clusters,
        }
        if verbose:
            print(f"  {tag}r={r:.2f}: clusters={n_clusters:3d}  "
                   f"mean ARI={mean_ari:.3f} ± {std_ari:.3f}")

    for col in (REF_KEY, SUB_KEY):
        if col in adata.obs.columns:
            del adata.obs[col]
    return scores, refs


def _build_null_adata(
    adata: anndata.AnnData,
    null_layer: Optional[str],
    null_seed: int,
    n_pcs: int,
    n_neighbors: int,
) -> anndata.AnnData:
    """Per-gene-permutation null per Lange et al. 2004.

    Independently shuffles each gene's expression vector across cells.
    This preserves per-gene marginal distributions (so PCA still
    captures *some* variance — chance variance) but destroys ALL
    cell-cell co-expression. PCA + kNN-graph + leiden on the result is
    the procedurally-matched null pipeline whose stability we subtract
    from the observed stability.
    """
    import scipy.sparse
    import scanpy as sc

    rng = np.random.default_rng(null_seed)
    if null_layer is not None and null_layer in adata.layers:
        X = adata.layers[null_layer]
    else:
        X = adata.X
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32).copy()
    # Independent per-gene permutation. Generator.shuffle operates
    # in-place along axis 0 by default — exactly per-column for a 2D array.
    for j in range(X.shape[1]):
        rng.shuffle(X[:, j])

    null_adata = anndata.AnnData(
        X=X,
        obs=adata.obs[[]].copy(),
        var=adata.var[[]].copy(),
    )
    # Use scanpy directly so the null pipeline doesn't touch ov.settings
    # dispatch — the null is a procedural twin, not a "best-effort
    # omicverse run". Keep PCA n_comps and neighbor count matched to the
    # real graph so the null is comparable.
    sc.pp.pca(null_adata, n_comps=n_pcs, zero_center=True)
    sc.pp.neighbors(null_adata, n_neighbors=n_neighbors,
                     use_rep='X_pca', random_state=0)
    return null_adata


# ───────────────────────────── public entry ───────────────────────────────────


@register_function(
    aliases=['自动分辨率选择', 'auto_resolution', 'autoResolution',
             'optimal leiden resolution'],
    category="single",
    description=(
        "Pick the most reproducible Leiden resolution by null-adjusted "
        "bootstrap-ARI stability (Lange et al. 2004). For each candidate "
        "resolution, mean-ARI stability is measured both on the real "
        "data and on a per-gene-permuted null with no cell-cell "
        "structure; the chosen resolution maximises (real − null)."
    ),
    prerequisites={'functions': ['pp.neighbors']},
    requires={'obsp': ['connectivities'], 'uns': ['neighbors']},
    produces={'obs': ['leiden'], 'uns': ['autoResolution']},
    auto_fix='auto',
    examples=[
        'res, df = ov.single.auto_resolution(adata)',
        'ov.single.auto_resolution(adata, resolutions=np.arange(0.2, 2.0, 0.1))',
    ],
    related=['pp.leiden'],
)
@tracked('auto_resolution', 'ov.single.auto_resolution')
def auto_resolution(
    adata: anndata.AnnData,
    resolutions: Optional[Sequence[float]] = None,
    *,
    method: str = 'bootstrap-ari',
    n_subsamples: int = 5,
    subsample_frac: float = 0.8,
    use_null_correction: bool = True,
    n_null_subsamples: int = 3,
    null_seed: int = 42,
    null_layer: Optional[str] = None,
    gamma_min: float = 0.05,
    gamma_max: float = 3.0,
    n_partitions: int = 30,
    min_clusters: int = 3,
    key_added: str = 'leiden',
    random_state: int = 0,
    verbose: bool = True,
):
    r"""Pick the most reproducible Leiden resolution via null-adjusted
    bootstrap-ARI (Lange, Roth, Braun & Buhmann, *Neural Computation*
    2004).

    Algorithm
    ---------
    For each candidate resolution :math:`r`:

    1. Run Leiden on the **full** AnnData → reference labels at ``r``.
    2. Take ``n_subsamples`` independent without-replacement
       subsamples of size ``subsample_frac × n_obs``.
    3. For each subsample run Leiden on the induced subgraph and
       compute Adjusted Rand Index against the reference restricted to
       the subsample.
    4. :math:`s_\mathrm{real}(r) = \mathrm{mean\,ARI}` across the
       ``n_subsamples`` bootstraps — the **observed**
       reproducibility.

    Bootstrap stability alone is biased toward fine resolutions: small
    tight clusters are mechanically reproducible under any subsampling
    procedure regardless of whether they reflect biological structure.
    The Lange–Buhmann fix is to subtract a procedurally-matched **null**:

    5. Build a null AnnData by independently permuting each gene's
       expression across cells. This preserves per-gene marginal
       distributions but destroys all cell-cell co-expression — there
       is no cluster structure left.
    6. Run the same PCA → kNN-graph → bootstrap-stability pipeline on
       the null with ``n_null_subsamples`` subsamples.
       :math:`s_\mathrm{null}(r)` is the chance-level reproducibility
       of leiden at this resolution given the data's marginal-only
       statistical structure.
    7. The **excess stability** is

       .. math::

           \Delta(r) = s_\mathrm{real}(r) - s_\mathrm{null}(r)

       and the chosen resolution is :math:`\arg\max_r \Delta(r)`
       subject to producing at least ``min_clusters`` clusters
       on the full data.

    Setting ``use_null_correction`` ``=False`` falls back to
    plain bootstrap-ARI; it's exposed mostly for diagnostics. The
    default is the null-adjusted variant.

    Parameters
    ----------
    adata
        AnnData with a precomputed neighbor graph
        (``adata.obsp['connectivities']``).
    resolutions
        Candidate resolutions to test. Defaults to
        ``np.round(np.arange(0.2, 1.6, 0.1), 2)``.
    n_subsamples
        Bootstrap subsamples per resolution on the **real** data.
    subsample_frac
        Fraction of cells in each subsample. Default 0.8.
    use_null_correction
        If ``True`` (default), subtract null-pipeline stability per
        Lange et al. 2004. ``False`` returns plain bootstrap stability.
    n_null_subsamples
        Bootstrap subsamples per resolution on the **null** data — can
        be smaller than ``n_subsamples`` because the null is
        low-variance.
    null_seed
        Seed for the per-gene permutation generating the null. Held
        separate from ``random_state`` so the null is
        reproducible independently of the real-data search.
    null_layer
        ``adata.layers`` key to permute. Defaults to ``adata.X``.
    min_clusters
        Lower bound on the number of clusters the chosen resolution
        must produce on the full data; degenerate resolutions are
        excluded from the argmax.
    key_added
        ``adata.obs`` column to write the chosen resolution's labels to.
    random_state
        Seed for subsample selection and Leiden on the real data.
    verbose
        Stream per-resolution scores during the search.

    Returns
    -------
    Tuple[anndata.AnnData, float, pandas.DataFrame]
        ``(adata, best_resolution, scores_df)``. ``scores_df`` is
        indexed by resolution with columns ``stability_real``,
        ``stability_null``, ``excess_stability``, ``std_real``,
        ``n_clusters``. Also writes ``adata.obs[key_added]`` and
        ``adata.uns['autoResolution']``.

    References
    ----------
    Lange, Roth, Braun, Buhmann. "Stability-based validation of
    clustering solutions." *Neural Computation* 16(6):1299–1323, 2004.

    Weir, Emmons, Wakefield, Hopkins, Mucha. "Post-processing partitions
    to identify domains of modularity optimization." *Algorithms* 10(3):93,
    2017 (used when ``method='champ'``).
    """
    # ── Route: method='champ' → deterministic modularity-geometric ──
    if method == 'champ':
        # Nested @tracked call — champ's own record_step is silenced
        # by the depth guard, so the entry will be owned by
        # auto_resolution and reflect that we used CHAMP.
        from ..pp._champ import champ as _champ
        result_adata, (gamma_lo, gamma_hi), df_champ = _champ(
            adata,
            resolutions=resolutions,
            n_partitions=n_partitions,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            key_added=key_added,
            random_state=random_state,
            verbose=verbose,
        )
        # Return the winning partition's origin_resolution as the
        # scalar ``best`` — the γ you'd plug into plain leiden to get
        # this same partition. The admissible range lives in
        # ``adata.uns[key_added]['chosen_gamma_range']``.
        widest = df_champ.loc[df_champ['gamma_range'].idxmax()]
        best_resolution = float(widest['origin_resolution'])
        note(
            backend=f'auto_resolution · CHAMP · '
                     f'γ ∈ [{gamma_lo:.3f}, {gamma_hi:.3f}] '
                     f'(width {gamma_hi - gamma_lo:.3f})',
            viz=[
                {'function': 'ov.pl.champ_landscape', 'kwargs': {}},
                {'function': 'ov.pl.cluster_sizes_bar',
                  'kwargs': {'groupby': key_added}},
                *([{'function': 'ov.pl.embedding',
                     'kwargs': {'basis': 'X_umap',
                                'color': key_added,
                                'frameon': 'small'}}]
                   if 'X_umap' in adata.obsm else []),
            ],
        )
        return result_adata, best_resolution, df_champ
    if method != 'bootstrap-ari':
        raise ValueError(
            f"method must be 'bootstrap-ari' (Lange 2004, default) or "
            f"'champ' (Weir 2017); got {method!r}."
        )

    # ── Default path: method='bootstrap-ari' ────────────────────────
    n_obs = adata.n_obs
    if n_obs < 50:
        raise ValueError(
            f"autoResolution needs at least 50 cells; got {n_obs}."
        )
    if 'connectivities' not in adata.obsp:
        raise ValueError(
            "autoResolution requires a precomputed neighbor graph "
            "(adata.obsp['connectivities']); run ov.pp.neighbors first."
        )
    if not (0.0 < subsample_frac < 1.0):
        raise ValueError(
            f"subsample_frac must be in (0, 1); got {subsample_frac}."
        )

    if resolutions is None:
        resolutions = list(np.round(np.arange(0.2, 1.6, 0.1), 2))
    resolutions = [float(np.round(r, 3)) for r in resolutions]

    rng = np.random.default_rng(random_state)
    real_subsamples = _subsample_indices(n_obs, n_subsamples,
                                          subsample_frac, rng)

    if verbose:
        msg = (f"{EMOJI['start']} autoResolution: testing "
                f"{len(resolutions)} resolutions × {n_subsamples} subsamples"
                f" (n_cells={n_obs})")
        if use_null_correction:
            msg += f" + null with {n_null_subsamples} subsamples"
        print(msg)

    # ── Real-data pass ─────────────────────────────────────────────
    real_scores, ref_labels = _bootstrap_stability_pass(
        adata, resolutions, real_subsamples,
        random_state=random_state,
        label='real' if use_null_correction else '', verbose=verbose,
    )

    # ── Null pass (per-gene permutation, Lange et al. 2004) ─────────
    if use_null_correction:
        n_pcs = (adata.obsm['X_pca'].shape[1]
                  if 'X_pca' in adata.obsm else 50)
        n_neighbors = (
            adata.uns.get('neighbors', {}).get('params', {})
                  .get('n_neighbors', 15)
        )
        if verbose:
            print(f"  building null AnnData "
                   f"(per-gene permutation, n_pcs={n_pcs}, "
                   f"n_neighbors={n_neighbors})...")
        null_adata = _build_null_adata(
            adata, null_layer, null_seed, n_pcs, n_neighbors,
        )
        null_rng = np.random.default_rng(random_state + 1)
        null_subsamples = _subsample_indices(
            null_adata.n_obs, n_null_subsamples, subsample_frac, null_rng,
        )
        null_scores, _ = _bootstrap_stability_pass(
            null_adata, resolutions, null_subsamples,
            random_state=random_state, label='null', verbose=verbose,
        )
    else:
        null_scores = {r: {'mean_ari': 0.0, 'std_ari': 0.0,
                              'n_clusters': real_scores[r]['n_clusters']}
                         for r in resolutions}

    # ── Combine ─────────────────────────────────────────────────────
    rows = []
    for r in resolutions:
        rs = real_scores[r]
        ns = null_scores[r]
        rows.append({
            'resolution':       r,
            'n_clusters':       rs['n_clusters'],
            'stability_real':   rs['mean_ari'],
            'stability_null':   ns['mean_ari'],
            'excess_stability': rs['mean_ari'] - ns['mean_ari'],
            'std_real':         rs['std_ari'],
        })
    df = pd.DataFrame(rows).set_index('resolution').sort_index()

    eligible = df[df['n_clusters'] >= min_clusters]
    if eligible.empty:
        raise RuntimeError(
            f"No resolution produced >= {min_clusters} clusters; consider "
            "lowering `min_clusters` or extending `resolutions`."
        )
    selector = 'excess_stability' if use_null_correction else 'stability_real'
    best = float(eligible[selector].idxmax())
    best_n_clusters = int(df.loc[best, 'n_clusters'])

    adata.obs[key_added] = pd.Categorical(ref_labels[best])
    adata.uns['autoResolution'] = {
        'best_resolution':     best,
        'scores':              df.reset_index().to_dict('list'),
        'n_subsamples':        int(n_subsamples),
        'n_null_subsamples':   int(n_null_subsamples) if use_null_correction else 0,
        'use_null_correction': bool(use_null_correction),
        'subsample_frac':      float(subsample_frac),
        'method': ('null-adjusted bootstrap-ARI (Lange et al. 2004)'
                    if use_null_correction else 'bootstrap-ARI'),
    }
    add_reference(
        adata, 'autoResolution',
        (f'auto-selected leiden resolution={best} via '
          f'{"null-adjusted " if use_null_correction else ""}ARI stability'),
    )

    if verbose:
        s_real = df.loc[best, 'stability_real']
        s_null = df.loc[best, 'stability_null']
        excess = df.loc[best, 'excess_stability']
        print(
            f"{EMOJI['done']} chosen resolution: {best} "
            f"({best_n_clusters} clusters; "
            f"real={s_real:.3f}, null={s_null:.3f}, excess={excess:.3f})"
        )

    note(
        backend=(f'omicverse · null-adjusted ARI · '
                  f'{n_subsamples}+{n_null_subsamples} subsamples'
                  if use_null_correction
                  else f'omicverse · ARI-stability · {n_subsamples} subsamples'),
        viz=[
            # Selection-curve so the report shows WHY this r was chosen.
            # Reads directly from adata.uns['autoResolution'].
            {'function': 'ov.pl.auto_resolution_curve', 'kwargs': {}},
            {'function': 'ov.pl.cluster_sizes_bar',
              'kwargs': {'groupby': key_added}},
            *([{'function': 'ov.pl.embedding',
                 'kwargs': {'basis': 'X_umap', 'color': key_added,
                            'frameon': 'small'}}]
               if 'X_umap' in adata.obsm else []),
        ],
    )

    return adata, best, df


# Backward-compatible camelCase alias. The canonical name is now
# auto_resolution; downstream code that still imports autoResolution
# keeps working.
autoResolution = auto_resolution
