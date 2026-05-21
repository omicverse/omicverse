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
                          ``simulate_airr``
Single-cell QC            ``chain_qc``
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
                          ``load_example_immdata``
B-cell / Ig analysis      ``clonal_clustering`` (identical / hierarchical /
                          spectral), ``distance_threshold``,
                          ``mutation_analysis``, ``shm_targeting``,
                          ``baseline_selection``, ``find_novel_alleles``,
                          ``infer_genotype`` (frequency / bayesian),
                          ``lineage_trees``, ``lineage_tests``
                          (switches / correlation), ``hill_diversity``,
                          ``aa_properties``
"""
from __future__ import annotations

import importlib as _importlib


# Lazy public surface — single source of truth for what's exposed.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # --- I/O (single-cell) ---
    "read_10x_vdj":              (".io", "read_10x_vdj"),
    "read_airr":                 (".io", "read_airr"),
    "read_tracer":               (".io", "read_tracer"),
    "simulate_airr":             (".io", "simulate_airr"),
    "airr_obs_columns":          (".io", "airr_obs_columns"),
    # --- single-cell QC ---
    "chain_qc":                  ("._qc", "chain_qc"),
    # --- clonotypes (single-cell) ---
    "ir_dist":                   ("._clonotype", "ir_dist"),
    "define_clonotypes":         ("._clonotype", "define_clonotypes"),
    "define_clonotype_clusters": ("._clonotype", "define_clonotype_clusters"),
    "clonal_expansion":          ("._clonotype", "clonal_expansion"),
    "clonotype_network":         ("._clonotype", "clonotype_network"),
    "clonotype_imbalance":       ("._clonotype", "clonotype_imbalance"),
    # --- repertoire metrics (single-cell) ---
    "alpha_diversity":           ("._metrics", "alpha_diversity"),
    "repertoire_overlap":        ("._metrics", "repertoire_overlap"),
    "group_abundance":           ("._metrics", "group_abundance"),
    "spectratype":               ("._metrics", "spectratype"),
    "vdj_usage":                 ("._metrics", "vdj_usage"),
    "clonotype_modularity":      ("._metrics", "clonotype_modularity"),
    # --- plotting ---
    "clonotype_network_plot":    (".plotting", "clonotype_network_plot"),
    "clonal_expansion_plot":     (".plotting", "clonal_expansion_plot"),
    "spectratype_plot":          (".plotting", "spectratype_plot"),
    "vdj_usage_plot":            (".plotting", "vdj_usage_plot"),
    "repertoire_overlap_plot":   (".plotting", "repertoire_overlap_plot"),
    "group_abundance_plot":      (".plotting", "group_abundance_plot"),
    # --- bulk repertoire (pyimmunarch) ---
    "repertoire_diversity":      ("._bulk", "repertoire_diversity"),
    "repertoire_overlap_bulk":   ("._bulk", "repertoire_overlap_bulk"),
    "gene_usage_bulk":           ("._bulk", "gene_usage_bulk"),
    "clonality":                 ("._bulk", "clonality"),
    "public_clonotypes":         ("._bulk", "public_clonotypes"),
    "track_clonotypes":          ("._bulk", "track_clonotypes"),
    "simulate_immdata":          ("._bulk", "simulate_immdata"),
    "load_example_immdata":      ("._bulk", "load_example_immdata"),
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
