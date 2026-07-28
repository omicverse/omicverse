"""
Immune-repertoire analysis for omicverse — a unified AIRR-seq (TCR / BCR) suite.

``ov.airr`` is the analogue of ``ov.protein`` / ``ov.genetics`` for adaptive
immune receptor repertoire sequencing.  It threads the major AIRR-seq
analyses behind one registered, dispatch-based API and spans two regimes:

**Single-cell AIRR analysis** — a clean, AnnData-native reimplementation of
the core of `scirpy <https://scirpy.scverse.org>`_.  Per-cell TCR / BCR
receptor data is stored in ``AnnData.obs`` (VJ_1 / VJ_2 / VDJ_1 / VDJ_2
chain slots), so single-cell repertoire analysis composes naturally with
the rest of omicverse's single-cell stack.

**Bulk + B-cell analysis** — thin wrappers over **six standalone
R-parity backend packages** that omicverse ships as separate releases:

* :mod:`pyimmunarch` — bulk repertoire: diversity / overlap / gene usage /
  clonality / public clonotypes / clonotype tracking.
* :mod:`pyalakazam`  — Immcantation core: Hill diversity, gene usage,
  CDR3 AA properties, sequence distances, lineage trees.
* :mod:`pyshazam`    — somatic hypermutation: distance-to-nearest /
  thresholds, targeting models, observed mutations, BASELINe selection.
* :mod:`pyscoper`    — B-cell clonal clustering: identical / hierarchical /
  spectral.
* :mod:`pytigger`    — immunoglobulin genotyping: novel-allele discovery,
  genotype inference, allele reassignment.
* :mod:`pydowser`    — B-cell phylogenetics: lineage trees, trait-switch
  tests, measurable-evolution tests.

Bulk / B-cell data is naturally tabular, so those functions take plain
AIRR-format :class:`pandas.DataFrame` objects (or the immunarch
:class:`pyimmunarch.ImmunData` container) rather than forcing everything
into AnnData — the single-cell side is AnnData-native.

Install the backends with::

    pip install omicverse[airr]

All backend imports are deferred to call-time — ``import omicverse.airr``
does no heavy work and succeeds even when no backend is installed.

Quick-start
-----------
>>> import omicverse as ov
>>> # --- single-cell TCR/BCR ---
>>> adata = ov.airr.read_10x_vdj('filtered_contig_annotations.csv')
>>> ov.airr.chain_qc(adata)
>>> adata = ov.airr.filter_chains(adata, keep='paired')  # keep dual-TCR cells
>>> ov.airr.define_clonotypes(adata)
>>> ov.airr.clonal_expansion(adata)
>>> ov.airr.clonotype_network(adata, min_cells=2)
>>> div = ov.airr.alpha_diversity(adata, groupby='sample')
>>> ov.airr.vdj_usage(adata, gene='v', groupby='group')
>>> # --- bulk repertoire (pyimmunarch) ---
>>> immdata = ov.airr.load_example_immdata()
>>> ov.airr.repertoire_diversity(immdata, method='chao1')
>>> # --- B-cell SHM + clonal clustering (Immcantation) ---
>>> db = ov.airr.clonal_clustering(bcr_db, method='hierarchical', threshold=0.15)
>>> ov.airr.mutation_analysis(db, frequency=True)

Pipeline stages
---------------
I/O                       ``read_10x_vdj``, ``read_airr``, ``read_tracer``,
                          ``from_airr_array``, ``simulate_airr``
Single-cell QC            ``chain_qc``, ``filter_chains``
Clonotypes (single-cell)  ``ir_dist``, ``define_clonotypes``,
                          ``define_clonotype_clusters``,
                          ``clonal_expansion``, ``clonotype_network``,
                          ``clonotype_imbalance``
Repertoire metrics        ``alpha_diversity``, ``repertoire_overlap``,
                          ``group_abundance``, ``spectratype``,
                          ``vdj_usage``, ``clonotype_modularity``
Plotting                  ``clonotype_network_plot``,
                          ``clonal_expansion_plot``, ``spectratype_plot``,
                          ``vdj_usage_plot``, ``repertoire_overlap_plot``,
                          ``group_abundance_plot``
Bulk repertoire           ``repertoire_diversity``,
                          ``repertoire_overlap_bulk``, ``gene_usage_bulk``,
                          ``clonality``, ``public_clonotypes``,
                          ``track_clonotypes``, ``simulate_immdata``,
                          ``load_example_immdata``, ``kmer_analysis``,
                          ``kmer_motif``, ``cdr3_aa_properties``,
                          ``gene_usage_analysis``, ``annotate_antigen_bulk``,
                          ``overlap_analysis``, ``public_repertoire``
B-cell / Ig analysis      ``clonal_clustering`` (identical / hierarchical /
                          spectral), ``distance_threshold``,
                          ``mutation_analysis``, ``shm_targeting``,
                          ``baseline_selection``, ``find_novel_alleles``,
                          ``infer_genotype`` (frequency / bayesian),
                          ``lineage_trees``, ``lineage_tests``
                          (switches / correlation), ``hill_diversity``,
                          ``aa_properties``, ``reconstruct_germlines``,
                          ``clonal_abundance``, ``bcr_gene_usage``
TCR specificity           ``tcrdist``, ``tcr_neighbors``, ``tcr_cluster``,
                          ``giana_cluster``, ``clustcr_cluster``,
                          ``meta_clonotypes``, ``specificity_groups``
                          (GLIPH2), ``annotate_antigen`` (VDJdb / McPAS /
                          IEDB), ``cdr3_logo``, ``cdr3_logo_background``,
                          ``detect_invariant`` (MAIT / iNKT)
TCR + GEX (CoNGA-style)   ``conga_score``, ``conga_clusters``,
                          ``tcr_clumping``, ``hotspot_features``
"""
from __future__ import annotations

import importlib as _importlib


# Lazy public surface — single source of truth for what's exposed.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # --- I/O (single-cell) ---
    "read_10x_vdj":              (".io", "read_10x_vdj"),
    "read_airr":                 (".io", "read_airr"),
    "read_tracer":               (".io", "read_tracer"),
    "from_airr_array":           (".io", "from_airr_array"),
    "extract_heavy_chains":      (".io", "extract_heavy_chains"),
    "simulate_airr":             (".io", "simulate_airr"),
    "airr_obs_columns":          (".io", "airr_obs_columns"),
    # --- single-cell QC ---
    "chain_qc":                  ("._qc", "chain_qc"),
    "filter_chains":             ("._qc", "filter_chains"),
    # --- clonotypes (single-cell) ---
    "ir_dist":                   ("._clonotype", "ir_dist"),
    "define_clonotypes":         ("._clonotype", "define_clonotypes"),
    "define_clonotype_clusters": ("._clonotype", "define_clonotype_clusters"),
    "clonal_expansion":          ("._clonotype", "clonal_expansion"),
    "clonal_expansion_composition": ("._clonotype", "clonal_expansion_composition"),
    "clonotype_network":         ("._clonotype", "clonotype_network"),
    "clonotype_imbalance":       ("._clonotype", "clonotype_imbalance"),
    # --- repertoire metrics (single-cell) ---
    "alpha_diversity":           ("._metrics", "alpha_diversity"),
    "repertoire_overlap":        ("._metrics", "repertoire_overlap"),
    "group_abundance":           ("._metrics", "group_abundance"),
    "spectratype":               ("._metrics", "spectratype"),
    "vdj_usage":                 ("._metrics", "vdj_usage"),
    "clonotype_modularity":      ("._metrics", "clonotype_modularity"),
    "summarize_by_group":        ("._metrics", "summarize_by_group"),
    "cluster_purity":            ("._metrics", "cluster_purity"),
    "label_agreement":           ("._metrics", "label_agreement"),
    "benchmark_clustering":      ("._metrics", "benchmark_clustering"),
    # --- plotting ---
    "clonotype_network_plot":    (".plotting", "clonotype_network_plot"),
    "clonal_expansion_plot":     (".plotting", "clonal_expansion_plot"),
    "spectratype_plot":          (".plotting", "spectratype_plot"),
    "vdj_usage_plot":            (".plotting", "vdj_usage_plot"),
    "repertoire_overlap_plot":   (".plotting", "repertoire_overlap_plot"),
    "group_abundance_plot":      (".plotting", "group_abundance_plot"),
    "kmer_motif_plot":           (".plotting", "kmer_motif_plot"),
    "cdr3_aa_profile_plot":      (".plotting", "cdr3_aa_profile_plot"),
    "gene_usage_analysis_plot":  (".plotting", "gene_usage_analysis_plot"),
    "clonal_abundance_plot":     (".plotting", "clonal_abundance_plot"),
    "group_box_plot":            (".plotting", "group_box_plot"),
    # --- bulk repertoire (pyimmunarch) ---
    "repertoire_diversity":      ("._bulk", "repertoire_diversity"),
    "repertoire_overlap_bulk":   ("._bulk", "repertoire_overlap_bulk"),
    "gene_usage_bulk":           ("._bulk", "gene_usage_bulk"),
    "clonality":                 ("._bulk", "clonality"),
    "public_clonotypes":         ("._bulk", "public_clonotypes"),
    "track_clonotypes":          ("._bulk", "track_clonotypes"),
    "simulate_immdata":          ("._bulk", "simulate_immdata"),
    "load_example_immdata":      ("._bulk", "load_example_immdata"),
    "kmer_analysis":             ("._bulk", "kmer_analysis"),
    "kmer_motif":                ("._bulk", "kmer_motif"),
    "cdr3_aa_properties":        ("._bulk", "cdr3_aa_properties"),
    "gene_usage_analysis":       ("._bulk", "gene_usage_analysis"),
    "annotate_antigen_bulk":     ("._bulk", "annotate_antigen_bulk"),
    "overlap_analysis":          ("._bulk", "overlap_analysis"),
    "public_repertoire":         ("._bulk", "public_repertoire"),
    "cohort_groups":             ("._bulk", "cohort_groups"),
    "repertoire_summary":        ("._bulk", "repertoire_summary"),
    "spectratype_bulk":          ("._bulk", "spectratype_bulk"),
    "differential_gene_usage":   ("._bulk", "differential_gene_usage"),
    "cdr3_aa_properties_by_group": ("._bulk", "cdr3_aa_properties_by_group"),
    "antigen_load_summary":      ("._bulk", "antigen_load_summary"),
    # --- B-cell / Ig analysis (Immcantation backends) ---
    "clonal_clustering":         ("._bcr", "clonal_clustering"),
    "distance_threshold":        ("._bcr", "distance_threshold"),
    "mutation_analysis":         ("._bcr", "mutation_analysis"),
    "shm_targeting":             ("._bcr", "shm_targeting"),
    "baseline_selection":        ("._bcr", "baseline_selection"),
    "find_novel_alleles":        ("._bcr", "find_novel_alleles"),
    "infer_genotype":            ("._bcr", "infer_genotype"),
    "lineage_trees":             ("._bcr", "lineage_trees"),
    "lineage_tests":             ("._bcr", "lineage_tests"),
    "hill_diversity":            ("._bcr", "hill_diversity"),
    "aa_properties":             ("._bcr", "aa_properties"),
    "reconstruct_germlines":     ("._bcr", "reconstruct_germlines"),
    "clonal_abundance":          ("._bcr", "clonal_abundance"),
    "bcr_gene_usage":            ("._bcr", "bcr_gene_usage"),
    "isotype_class":             ("._bcr", "isotype_class"),
    "isotype_composition":       ("._bcr", "isotype_composition"),
    "clone_timepoint_distribution": ("._bcr", "clone_timepoint_distribution"),
    "mutation_by_region":        ("._bcr", "mutation_by_region"),
    "collapse_germlines":        ("._bcr", "collapse_germlines"),
    "normalize_gene_calls":      ("._bcr", "normalize_gene_calls"),
    # --- TCR specificity (TCRdist / GLIPH2 / meta-clonotypes) ---
    "tcrdist":                   ("._tcr", "tcrdist"),
    "tcr_neighbors":             ("._tcr", "tcr_neighbors"),
    "tcr_cluster":               ("._tcr", "tcr_cluster"),
    "giana_cluster":             ("._tcr", "giana_cluster"),
    "clustcr_cluster":           ("._tcr", "clustcr_cluster"),
    "meta_clonotypes":           ("._tcr", "meta_clonotypes"),
    "specificity_groups":        ("._tcr", "specificity_groups"),
    "annotate_antigen":          ("._tcr", "annotate_antigen"),
    "cdr3_logo":                 ("._tcr", "cdr3_logo"),
    "cdr3_logo_background":      ("._tcr", "cdr3_logo_background"),
    "detect_invariant":          ("._tcr", "detect_invariant"),
    "clean_cdr3":                ("._tcr", "clean_cdr3"),
    "usable_cdr3_mask":          ("._tcr", "usable_cdr3_mask"),
    "tcrdist_embedding":         ("._tcr", "tcrdist_embedding"),
    "specificity_group_purity":  ("._tcr", "specificity_group_purity"),
    # --- TCR + GEX joint analysis (CoNGA-style) ---
    "conga_score":               ("._tcr_gex", "conga_score"),
    "conga_clusters":            ("._tcr_gex", "conga_clusters"),
    "tcr_clumping":              ("._tcr_gex", "tcr_clumping"),
    "hotspot_features":          ("._tcr_gex", "hotspot_features"),
    "conga_cluster_table":       ("._tcr_gex", "conga_cluster_table"),
    "conga_score_summary":       ("._tcr_gex", "conga_score_summary"),
    "expression_by_group":       ("._tcr_gex", "expression_by_group"),
    "conga_score_plot":          ("._tcr_gex", "conga_score_plot"),
    "tcr_clumping_plot":         ("._tcr_gex", "tcr_clumping_plot"),
    "hotspot_features_plot":     ("._tcr_gex", "hotspot_features_plot"),
}

_LAZY_SUBMODULES = {"io", "plotting"}

_REGISTRY_SUBMODULES = (
    ".io",
    "._qc",
    "._clonotype",
    "._metrics",
    ".plotting",
    "._bulk",
    "._bcr",
    "._tcr",
    "._tcr_gex",
)


def _hydrate_registry() -> None:
    """Force-import every @register_function-bearing submodule so the global
    registry sees ov.airr at export time. Called from
    :mod:`omicverse._registry._hydrate_registry_for_export`."""
    for mod in _REGISTRY_SUBMODULES:
        try:
            _importlib.import_module(mod, __name__)
        except Exception:
            # Optional backends (pyimmunarch, pyscoper, …) may be missing —
            # register whatever loads cleanly.
            continue


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys())
                      + list(_LAZY_ATTRS.keys())
                      + list(_LAZY_SUBMODULES)))


__version__ = "0.1.0"

__all__ = list(_LAZY_ATTRS.keys()) + list(_LAZY_SUBMODULES)
