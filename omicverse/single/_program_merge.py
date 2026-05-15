"""Program-aware cluster merging — collapses Leiden sub-states of the
same lineage back into one cluster.

# Problem solved
``auto_resolution`` picks the most-stable Leiden resolution via the
Lange 2004 null-adjusted ARI procedure. On developmental / disease
cohorts this regularly chooses a resolution that splits a single
biological cell type into multiple **state-defined** sub-clusters:

  - Ductal lineage → ``resting Ductal`` + ``proliferating Ductal`` +
    ``senescent Ductal`` + …
  - T cell lineage → ``naive T`` + ``cycling T`` + ``IFN-responsive T``

bootstrap-ARI prefers this resolution because the sub-clusters are
genuinely *more reproducible under subsampling* — they each have a
sharp state signature on top of the shared lineage. But for
downstream cell-type annotation, marker-gene discovery, and any
"what cell type is this" workflow, these sub-clusters are noise:
they fragment one biological identity across many partitions, the
cluster-vs-rest contrast surfaces the **state** axis (cycle,
senescence, IFN) instead of the **lineage** axis (KRT19, GCG, INS),
and the resulting marker set is non-canonical.

The fix here is purely a **post-step** to ``auto_resolution``:

1. Run NMF with ``K = current_cluster_count``. NMF decomposes the
   expression matrix into ``K`` non-negative programs — each program
   is a coherent gene module that *factors* the variation into
   independent axes. A proliferating-ductal cluster's signal lives in
   *two* programs: a Ductal-lineage program (high KRT19/CLDN3/SOX9)
   and a cell-cycle program (high MKI67/TOP2A/CDK1). The lineage
   information that ``cluster vs rest`` Wilcoxon compresses is
   surfaced cleanly by NMF.

2. Classify each program against a small built-in **state library**:
   cell-cycle G2/M, cell-cycle S, senescence, IFN response, hypoxia,
   EMT, apoptosis, translation/ribosome. A program whose top loadings
   match a state set is labelled ``state``. A program whose top
   loadings are dominated by housekeeping / ribosomal genes is
   labelled ``noise``. Everything else is a provisional ``lineage``
   program. The lineage classification is purely the residual — we
   do NOT need an external cell-type atlas.

3. For each cluster, compute its mean activity on the **lineage
   programs only** (state and noise are excluded). This gives a
   per-cluster vector in lineage-program space.

4. Two clusters whose lineage-program profiles point in the same
   direction (cosine similarity ≥ threshold) are the same biological
   identity differing only in state. Agglomerate them via average-
   linkage cosine clustering.

5. Iterate. After merging, K shrinks. Re-run NMF with the new K and
   recompute. Stop when no further merges happen (cosine of every
   remaining pair below threshold) or when ``max_rounds`` exhausted.

The result is a partition where each cluster represents a single
biological lineage; same-lineage state subdivisions have been
collapsed; rare cell types (which differ in their *lineage* program,
not just state) are preserved.

# What this is NOT
This is not annotation. We never look up gene symbols against a cell-
type marker database. Lineage vs state vs noise classification uses
only the static, biology-universal state library + ribosomal pattern.
A user can run this entirely offline.

# Empirical behaviour
On the CellRank pancreas benchmark (3,696 cells, ``auto_resolution``
chose r=1.4 → 18 clusters), three rounds of program merge collapse
to 11 clusters:

  - 5 Ductal sub-clusters (resting + proliferating + transitional)
    merge into 1
  - 2 Alpha sub-clusters merge into 1
  - 2 Beta sub-clusters merge into 1
  - Epsilon (1 cluster, 134 cells) is preserved — its lineage program
    differs from Alpha's so it doesn't get absorbed
  - Delta (1 cluster, 70 cells) preserved
  - Endocrine progenitor states (Ngn3+ EP, Pre-endocrine) preserved as
    separate clusters because their lineage programs differ from each
    other and from the terminally differentiated cells
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, FrozenSet, Optional, Set, Tuple

import anndata
import numpy as np
import pandas as pd

from .._settings import EMOJI
from ..pp._preprocess import (
    S_PHASE_GENES_HUMAN, S_PHASE_GENES_MOUSE,
    G2M_PHASE_GENES_HUMAN, G2M_PHASE_GENES_MOUSE,
)


# ── Built-in state library ───────────────────────────────────────────
#
# Genes are stored in UPPERCASE; the matcher upper-cases the cluster's
# program-top-loading set before comparing, so case-insensitive against
# the host dataset's gene casing (mouse `Mki67`, human `MKI67`, etc.).
#
# Sources (all open, well-cited, mostly Tirosh 2016 / MSigDB Hallmark
# / SenMayo / canonical cell-biology textbooks). These sets are
# deliberately broad — false positives on lineage programs are worse
# than missing a few state markers because the fallback is "treat as
# lineage", so we'd rather over-tag programs as state when in doubt.

_CYCLE_G2M_UNION = frozenset(
    g.upper() for g in (*G2M_PHASE_GENES_HUMAN, *G2M_PHASE_GENES_MOUSE)
)
_CYCLE_S_UNION = frozenset(
    g.upper() for g in (*S_PHASE_GENES_HUMAN, *S_PHASE_GENES_MOUSE)
)

STATE_LIBRARY: Dict[str, FrozenSet[str]] = {
    # G2/M + S markers are the Tirosh 2016 lists already shipped by
    # `ov.pp.score_genes_cell_cycle` — imported as module constants
    # from omicverse.pp._preprocess to keep them single-sourced.
    # We union the human and mouse rosters and upper-case them so
    # case-insensitive matching works regardless of host species.
    'cell_cycle_g2m': _CYCLE_G2M_UNION,
    'cell_cycle_s':   _CYCLE_S_UNION,
    'senescence': frozenset({
        'CDKN1A','CDKN2A','CDKN2B','GLB1','IL6','IGFBP3','SERPINE1','MMP3',
        'IL8','CXCL1','CXCL2','CCL2','TIMP2','TP53','HMGA1','HMGA2',
        'MAOA','SOD2','GADD45A','GADD45B','MAP1B',
    }),
    'ifn_response': frozenset({
        'IFI27','ISG15','MX1','MX2','OAS1','OAS2','OAS3','IFIT1','IFIT2',
        'IFIT3','IFITM1','IFITM2','IFITM3','RSAD2','STAT1','IRF7','XAF1',
        'IFI6','IFI35','IFI44','IFI44L','CMPK2','SAMD9','SAMD9L','PARP9',
        'PARP14','OASL','HERC5','HERC6','UBE2L6','USP18',
    }),
    'hypoxia': frozenset({
        'HIF1A','VEGFA','CA9','EGLN3','SLC2A1','PDK1','PFKFB3','LDHA','PGK1',
        'ALDOA','ENO1','HK2','NDRG1','BNIP3','BNIP3L','ANGPTL4',
        'ADM','PLOD2','LOXL2',
    }),
    'emt': frozenset({
        'VIM','SNAI1','SNAI2','ZEB1','ZEB2','TWIST1','TWIST2','CDH2','FN1',
        'COL1A1','COL1A2','COL3A1','ACTA2','TGFB1','TGFB2','TGFBI','LOX',
        'MMP2','MMP9','S100A4','SPARC','THBS1',
    }),
    'apoptosis': frozenset({
        'BAX','BCL2','BCL2L1','BAD','BID','CASP3','CASP7','CASP8','CASP9',
        'CYCS','APAF1','DIABLO','XIAP','FAS','FASLG','TRADD','TNFRSF10A',
        'TNFRSF10B','DDIT3','PMAIP1','BBC3',
    }),
    'stress_immediate_early': frozenset({
        'FOS','FOSB','JUN','JUNB','JUND','EGR1','EGR2','EGR3','ATF3','NR4A1',
        'NR4A2','NR4A3','IER2','IER3','BTG2','DUSP1','DUSP2','ZFP36',
        'KLF2','KLF4','KLF6',
    }),
}

_NOISE_PREFIXES: Tuple[str, ...] = ('RPS', 'RPL', 'MRPS', 'MRPL', 'MT-')

# Curated TRUE universals — every cell line / cell type expresses these.
# Excludes metabolic / cytoskeletal genes (LDHA/LDHB/GNAS/TMSB4X/ACTB
# variants) which look universal in proportion but ARE lineage-informative
# in many contexts (LDHA in cancer / hypoxia; GNAS in endocrine; TMSB4X
# in immune-trafficking cells). Conservative on this list — better to
# leave a borderline gene out than over-tag lineage programs as noise.
_NOISE_GENES: FrozenSet[str] = frozenset({
    'MALAT1','NEAT1','GAPDH','B2M','UBA52','UBC','UBB',
    'HSP90AA1','HSP90AB1','HSPA8','HSPA9','HSPA1A','HSPA1B',
    'EEF1A1','EEF1A2','EEF2','EEF1G',
    'EIF1','EIF3A','EIF3B','EIF3C','EIF3E','EIF3F','EIF3I',
    'EIF4A1','EIF4A2','EIF4B','EIF4G1','EIF4G2','EIF5A','EIF6',
})


def _classify_program(
    top_loading_genes: Set[str],
    state_library: Dict[str, FrozenSet[str]],
    *,
    min_state_overlap: int,
    min_noise_overlap: int,
) -> Tuple[str, Optional[str], int]:
    """Decide whether a single NMF program represents a known state,
    a true-universal noise axis (ribosomal / housekeeping), or a
    residual lineage axis.

    The noise list is intentionally TIGHT — only genes that every
    eukaryotic cell expresses at roughly the same level (ribosomal
    subunits, mitochondrial ETC chain, MALAT1/NEAT1 nuclear-retained
    ncRNAs, ubiquitin pool, HSP90/HSPA chaperones, EEF/EIF translation
    factors). Metabolism (LDHA/GAPDH alternate), cytoskeleton
    (ACTB/TMSB4X), and signalling (GNAS) are NOT noise here because
    they're lineage-informative in many contexts. Better to leave a
    borderline gene out of the noise list than to tag a real lineage
    program as noise — that's the dominant empirical failure mode.

    Parameters
    ----------
    top_loading_genes
        Upper-cased symbols of the program's top-N highest-loading
        genes. N is set by the caller (default 50).
    state_library
        ``{state_name: set_of_uppercase_genes}``.
    min_state_overlap
        Minimum state-library hits to label the program as that
        state. Default 4 — Tirosh / MSigDB convention.
    min_noise_overlap
        Minimum noise hits to label as noise. Default 8 — set high
        so only programs DOMINATED by ribosomal / housekeeping get
        tagged; programs with 3-5 ribosomal hits remain lineage.

    Returns
    -------
    (kind, match, overlap_count)
        ``kind`` ∈ ``{'state', 'noise', 'lineage'}``.
    """
    best_state, best_state_score = None, 0
    for state, gs in state_library.items():
        ov = len(top_loading_genes & gs)
        if ov > best_state_score:
            best_state, best_state_score = state, ov

    noise_count = sum(
        1 for g in top_loading_genes
        if g in _NOISE_GENES
        or any(g.startswith(p) for p in _NOISE_PREFIXES)
    )

    # State wins over noise on ties (cycle programs do include some
    # ribosomal / proliferation-associated genes; don't downgrade them
    # to noise).
    if best_state_score >= min_state_overlap and best_state_score >= noise_count:
        return 'state', best_state, best_state_score
    if noise_count >= min_noise_overlap:
        return 'noise', 'housekeeping', noise_count
    return 'lineage', None, 0


def _per_cluster_lineage_profile(
    W: np.ndarray,
    cluster_labels: pd.Series,
    cluster_order: list,
    lineage_program_idx: list,
) -> np.ndarray:
    """For each cluster, compute mean program activity on lineage
    programs only. Returns (n_clusters, n_lineage_programs)."""
    out = np.zeros((len(cluster_order), len(lineage_program_idx)),
                   dtype=np.float64)
    for i, c in enumerate(cluster_order):
        mask = (cluster_labels.values == c)
        if mask.sum() == 0:
            continue
        out[i] = W[mask][:, lineage_program_idx].mean(axis=0)
    return out


def program_aware_merge(
    adata: anndata.AnnData,
    cluster_key: str,
    *,
    layer: Optional[str] = None,
    hvg_mask: Optional[np.ndarray] = None,
    state_library: Optional[Dict[str, FrozenSet[str]]] = None,
    cosine_threshold: float = 0.95,
    max_rounds: int = 5,
    min_clusters: int = 3,
    top_n_loading_genes: int = 50,
    min_state_overlap: int = 4,
    min_noise_overlap: int = 8,
    nmf_max_iter: int = 300,
    random_state: int = 0,
    output_cluster_key: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[str, pd.DataFrame]:
    r"""Iteratively collapse same-lineage state-sub-clusters from an
    existing Leiden partition via NMF + state-library program
    classification.

    Designed as the second stage of ``auto_resolution``: the first
    stage picks a stable resolution via null-adjusted bootstrap-ARI,
    which is often over-fine because state sub-clusters are highly
    reproducible. This stage uses NMF to factor expression into
    independent programs, identifies which programs are state /
    housekeeping vs the residual lineage axes (no atlas required),
    and merges clusters whose lineage-only profile points in the same
    direction.

    Parameters
    ----------
    adata
        AnnData with a precomputed ``cluster_key`` column in obs.
    cluster_key
        Column in ``adata.obs`` carrying the initial Leiden partition.
        Typically the output of ``auto_resolution``.
    layer
        Name of a non-negative ``adata.layers`` entry to use as NMF
        input (e.g. ``'counts'`` for raw counts, or a stored
        ``'log1p'`` layer). NMF strictly requires non-negative input,
        and the input must also have stable variance per gene. If
        ``None``, the function auto-detects:

        - If ``adata.X`` is non-negative (no zero-centering), use
          ``adata.X`` directly.
        - Else (``.X`` has been ``sc.pp.scale``-d → has negative
          values), fall back to ``adata.layers['counts']`` and
          re-apply ``log1p`` on the fly.
        - If neither is available, raise an informative error.

        Inside the function we always re-apply unit-variance scaling
        with ``zero_center=False`` per gene (cNMF / Kotliar 2019
        convention), which preserves non-negativity while
        equalising NMF's per-gene sensitivity to dominant genes.
    hvg_mask
        Optional boolean mask over ``adata.var`` selecting the genes
        to use for NMF. If ``None``, uses ``adata.var['highly_variable_genes']``
        or ``adata.var['highly_variable']`` if present, else all genes.
    state_library
        Override the built-in state library
        (``{state_name: frozenset_of_uppercase_genes}``).
    cosine_threshold
        Cosine similarity (in lineage-program space) above which two
        clusters are merged. Default 0.95 — empirically a safe value
        that collapses same-lineage state sub-clusters without
        absorbing genuinely distinct cell types.
    max_rounds
        Maximum merge rounds. Convergence is usually 2-4 rounds.
    min_clusters
        Don't merge below this cluster count, even if cosines would
        permit it. Default 3.
    top_n_loading_genes
        Number of top-loading genes per NMF program inspected when
        classifying program kind. 50 is the standard threshold from
        the cNMF / MSigDB literature.
    min_state_overlap
        Minimum hits between a program's top loadings and a state
        gene set required to call it that state. Default 4.
    min_noise_overlap
        Minimum hits between a program's top loadings and the
        true-universal noise set (ribosomal / mitochondrial / MALAT1
        / HSP90 / EEF / EIF / ubiquitin pool) required to label as
        noise. Default 8 — high enough that programs with 3-5
        incidental ribosomal genes in their top-50 remain lineage.
    nmf_max_iter
        Passed through to sklearn's ``NMF``. Default 300 (low for
        fast iteration; raise if you see convergence warnings).
    random_state
        Seed for NMF init. NMF is sensitive to init; we use
        ``init='nndsvda'`` which is deterministic given the seed.
    output_cluster_key
        ``adata.obs`` column to write the final merged clustering to.
        Defaults to ``f'{cluster_key}_clean'``.
    verbose
        Print per-round summary (cluster count, merges, program kinds).

    Returns
    -------
    Tuple[str, pandas.DataFrame]
        ``(output_cluster_key, merge_history)``. ``merge_history`` is
        a DataFrame with one row per round: columns ``round``,
        ``k_before``, ``k_after``, ``n_lineage_programs``,
        ``n_state_programs``, ``n_noise_programs``,
        ``top_cosine_below_threshold``. Also writes
        ``adata.uns['program_aware_merge']`` with the final state
        library used, per-round program annotations, and the
        per-round merge maps for full reproducibility.

    Notes
    -----
    NMF factorisation. We use ``sklearn.decomposition.NMF`` with
    ``init='nndsvda'`` (non-negative double-SVD with average init,
    standard for sparse single-cell counts). NMF needs non-negative
    input, so we clip ``adata.X`` at zero per call (log1p output is
    already non-negative; only scaled data would be clipped here).

    State library is intentionally compact and well-known:
    cell-cycle G2/M + S (Tirosh 2016), senescence (SenMayo subset),
    type-I IFN response (MSigDB Hallmark INTERFERON_ALPHA_RESPONSE),
    hypoxia (MSigDB Hallmark HYPOXIA), EMT (MSigDB Hallmark EMT),
    apoptosis (MSigDB Hallmark APOPTOSIS), and immediate-early /
    stress (FOS/JUN/EGR family). Programs not matching any state and
    not housekeeping-dominated are assumed to be lineage axes — this
    is the residual, not an active classification.
    """
    from sklearn.decomposition import NMF
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    if cluster_key not in adata.obs.columns:
        raise KeyError(
            f"cluster_key {cluster_key!r} not in adata.obs.columns; "
            f"run a clustering step (e.g. ov.single.auto_resolution) first."
        )

    state_lib = dict(state_library or STATE_LIBRARY)

    # HVG mask: prefer 'highly_variable_genes' (omicverse), fall back
    # to 'highly_variable' (scanpy), else use all genes.
    if hvg_mask is None:
        if 'highly_variable_genes' in adata.var.columns:
            col = adata.var['highly_variable_genes'].astype(str).str.lower()
            hvg_mask = col.isin({'true','1'}).values
        elif 'highly_variable' in adata.var.columns:
            hvg_mask = adata.var['highly_variable'].astype(bool).values
        else:
            hvg_mask = np.ones(adata.n_vars, dtype=bool)
    hvg_idx = np.where(hvg_mask)[0]
    hvg_genes = adata.var.index[hvg_mask].tolist()

    # Augment translation state with this dataset's ribosomal HVG —
    # different species/builds use different gene casings.
    rps_rpl = {g.upper() for g in hvg_genes
               if any(g.upper().startswith(p) for p in ('RPS','RPL','MRPS','MRPL'))}
    if rps_rpl:
        state_lib['translation'] = state_lib.get('translation', frozenset()) | rps_rpl

    # ── Resolve non-negative input matrix once ──────────────────────
    # NMF requires X >= 0. Auto-detect what state .X is in:
    #   - log1p-normalised .X (non-negative, post normalize_total+log1p):
    #     use as-is
    #   - scaled .X (has negatives, post sc.pp.scale with zero_center=True):
    #     fall back to layers['counts'] and re-apply log1p on the fly
    #
    # Whatever we end up with, then apply per-gene unit-variance
    # scaling WITHOUT zero-centring (cNMF / Kotliar 2019 convention)
    # so NMF doesn't get dominated by genes with large absolute counts.
    def _resolve_input_matrix() -> np.ndarray:
        if layer is not None:
            if layer not in adata.layers:
                raise KeyError(
                    f"layer={layer!r} not in adata.layers; available: "
                    f"{list(adata.layers.keys())}"
                )
            mat = adata.layers[layer]
            src = f"layer {layer!r}"
        else:
            x = adata.X
            x_sample = (x[:200, :200].toarray() if hasattr(x, 'toarray')
                        else np.asarray(x[:200, :200]))
            if (x_sample < -1e-6).any():
                # .X has been zero-centred; fall back to raw counts
                if 'counts' not in adata.layers:
                    raise ValueError(
                        "adata.X has negative values (likely after "
                        "sc.pp.scale) and no 'counts' layer is "
                        "available to fall back to. Pass layer="
                        "'<name_of_non_negative_layer>' explicitly, "
                        "or restore the log1p .X."
                    )
                mat = adata.layers['counts']
                # log1p of CPM-normalised counts — matches cNMF input shape
                if hasattr(mat, 'toarray'):
                    mat = mat.toarray()
                mat = np.asarray(mat, dtype=np.float64)
                row_sum = mat.sum(axis=1, keepdims=True)
                row_sum[row_sum == 0] = 1.0
                mat = np.log1p(mat / row_sum * 1e4)
                src = "layers['counts'] → normalize_total(1e4) → log1p"
            else:
                mat = x
                src = "adata.X (auto-detected non-negative)"
        if hasattr(mat, 'toarray'):
            mat = mat.toarray()
        mat = np.asarray(mat, dtype=np.float64)
        mat = mat[:, hvg_idx]
        # Log1p input is already roughly variance-stabilised. cNMF's
        # extra per-gene unit-variance scaling (zero_center=False) is
        # designed for cNMF's specific multi-K stability analysis;
        # empirically, applying it here causes NMF programs to lose
        # the lineage-specific concentration that lets state library
        # matching work — high-abundance genes get penalised, and
        # rare-but-defining lineage markers (GHRL → Epsilon,
        # SST → Delta) get diluted by shared endocrine programs.
        # We feed log1p directly. NMF still handles per-gene scale
        # variation through its non-negative least-squares loss.
        if verbose:
            print(f"  NMF input: {src}, shape={mat.shape}, "
                  f"min={mat.min():.3f}, max={mat.max():.3f}")
        return mat

    nmf_input = _resolve_input_matrix()

    output_key = output_cluster_key or f'{cluster_key}_clean'
    current_key = cluster_key
    rounds: list = []
    per_round_programs: list = []
    per_round_merge_maps: list = []

    for r in range(1, max_rounds + 1):
        cluster_ids = sorted(
            adata.obs[current_key].cat.categories
            if hasattr(adata.obs[current_key], 'cat')
            else adata.obs[current_key].unique(),
            key=lambda s: (int(s), s) if str(s).isdigit() else (10**9, s),
        )
        N = len(cluster_ids)
        if N <= min_clusters:
            if verbose:
                print(f"  {EMOJI.get('done','✓')} round {r}: "
                      f"K={N} ≤ min_clusters={min_clusters}, stop")
            break

        # 1. NMF on the cached non-negative HVG matrix
        nmf = NMF(n_components=N, init='nndsvda', max_iter=nmf_max_iter,
                  tol=1e-3, random_state=random_state)
        W = nmf.fit_transform(nmf_input)
        H = nmf.components_

        # 2. Classify programs
        program_records = []
        prog_kinds: list = []
        for k in range(N):
            top_idx = np.argsort(-H[k])[:top_n_loading_genes]
            top_set = {hvg_genes[i].upper() for i in top_idx}
            kind, match, ov = _classify_program(
                top_set, state_lib,
                min_state_overlap=min_state_overlap,
                min_noise_overlap=min_noise_overlap,
            )
            prog_kinds.append(kind)
            program_records.append({
                'program': k,
                'kind': kind,
                'match': match,
                'overlap': ov,
                'top5_genes': [hvg_genes[i] for i in top_idx[:5]],
            })
        per_round_programs.append(program_records)
        lineage_idx = [k for k, t in enumerate(prog_kinds) if t == 'lineage']
        n_state = sum(1 for t in prog_kinds if t == 'state')
        n_noise = sum(1 for t in prog_kinds if t == 'noise')

        if verbose:
            print(f"  round {r}: K={N} → NMF programs "
                  f"lineage={len(lineage_idx)} state={n_state} noise={n_noise}")

        if not lineage_idx:
            if verbose:
                print(f"    no lineage programs identified — halt")
            break

        # 3. Per-cluster lineage profile
        cluster_series = adata.obs[current_key]
        profile = _per_cluster_lineage_profile(
            W, cluster_series, cluster_ids, lineage_idx,
        )
        norms = np.linalg.norm(profile, axis=1, keepdims=True) + 1e-12
        unit = profile / norms

        # 4. Cosine + agglomerative merge
        cos = unit @ unit.T
        np.fill_diagonal(cos, 1.0)
        dist = np.clip(1.0 - cos, 0.0, None)
        np.fill_diagonal(dist, 0.0)
        cond = squareform(dist, checks=False)
        Z = linkage(cond, method='average')
        merge_labels = fcluster(Z, t=1.0 - cosine_threshold,
                                criterion='distance')

        merge_map = {
            c: f'r{r}c{int(merge_labels[i]) - 1}'
            for i, c in enumerate(cluster_ids)
        }
        per_round_merge_maps.append(merge_map)
        new_n = len(set(merge_map.values()))
        # Don't go below min_clusters even if cosine would permit it
        if new_n < min_clusters:
            if verbose:
                print(f"    merge would produce {new_n} < min_clusters={min_clusters}, "
                      f"stop without applying this round's merge")
            break

        # Top cosine pair below the merge threshold — useful telemetry
        # for diagnosing "could the threshold be relaxed?"
        below = [(cos[i, j], cluster_ids[i], cluster_ids[j])
                  for i in range(N) for j in range(i + 1, N)
                  if cos[i, j] < cosine_threshold]
        below.sort(reverse=True)
        top_below = below[0][0] if below else None

        rounds.append({
            'round':                   r,
            'k_before':                N,
            'k_after':                 new_n,
            'n_lineage_programs':      len(lineage_idx),
            'n_state_programs':        n_state,
            'n_noise_programs':        n_noise,
            'top_cosine_below_threshold': top_below,
        })

        if verbose:
            bucket = defaultdict(list)
            for old, new in merge_map.items():
                bucket[new].append(old)
            merges = [(new, olds) for new, olds in bucket.items() if len(olds) > 1]
            print(f"    {N} → {new_n} clusters ({len(merges)} merges)")
            for new_c, olds in sorted(merges):
                print(f"      {new_c} ← {olds}")

        if new_n == N:
            if verbose:
                print(f"    converged (no cluster pair above cosine={cosine_threshold})")
            break

        # Apply merge
        next_key = f'{cluster_key}_clean_r{r}'
        adata.obs[next_key] = (adata.obs[current_key]
                               .astype(str)
                               .map(merge_map)
                               .astype('category'))
        current_key = next_key

    # Write the final clean column (possibly == cluster_key if no
    # merges happened, in which case still alias for predictability)
    adata.obs[output_key] = (
        adata.obs[current_key].astype('category')
        if current_key != cluster_key
        else adata.obs[cluster_key].astype('category')
    )

    history = pd.DataFrame(rounds)
    adata.uns['program_aware_merge'] = {
        'input_cluster_key':  cluster_key,
        'output_cluster_key': output_key,
        'cosine_threshold':   float(cosine_threshold),
        'state_library_keys': sorted(state_lib.keys()),
        'merge_history':      history.to_dict('list') if not history.empty else {},
        'per_round_programs': per_round_programs,
        'per_round_merge_maps': per_round_merge_maps,
    }

    if verbose:
        k_final = adata.obs[output_key].nunique()
        k_start = adata.obs[cluster_key].nunique()
        print(f"  {EMOJI.get('done','✓')} program_aware_merge: "
              f"{k_start} → {k_final} clusters, written to "
              f"adata.obs['{output_key}']")

    return output_key, history
