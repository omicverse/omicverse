"""
Metabolomics analysis for omicverse — AnnData-native peak-table workflows.

``ov.metabol`` covers the downstream-of-peak-picking stages of a
metabolomics study: QC, imputation, normalization, transformation,
univariate and multivariate statistics, pathway enrichment (MSEA /
mummichog), and a lipidomics path. Input is a peak table (produced by
XCMS / MZmine / MS-DIAL / OpenMS / MetaboAnalyst) loaded into an
AnnData with ``obs=samples``, ``var=metabolites`` — the same
convention as every other omicverse module.

Quick-start
-----------
>>> from omicverse.metabol import pyMetabo, read_metaboanalyst
>>> # group_col is required — pass the factor column name from your CSV
>>> adata = read_metaboanalyst("human_cachexia.csv", group_col="Muscle loss")
>>> m = pyMetabo(adata)
>>> (m.impute(method="qrilc", seed=0)
...    .normalize(method="pqn")
...    .transform(method="log")
...    .differential(method="welch_t", log_transformed=True)
...    .transform(method="pareto", stash_raw=False)
...    .opls_da(n_ortho=1))
>>> m.deg_table.head()
>>> m.vip_table().head()

Pipeline stages
---------------
I/O                      ``read_metaboanalyst``, ``read_wide``, ``read_lcms``
QC (MS-specific)         ``cv_filter``, ``drift_correct``, ``blank_filter``, ``sample_qc`` (Hotelling T² + DModX)
Batch / drift            ``serrf`` (QC-based Random Forest, Fan 2019)
Imputation               ``impute`` (kNN / half-min / QRILC / zero)
Sample normalization     ``normalize`` (PQN / TIC / median / MSTUS)
Feature transform        ``transform`` (log / glog / autoscale / Pareto)
Univariate differential  ``differential`` (2 groups) / ``anova`` (3+ groups Welch/ANOVA/Kruskal)
Multi-factor designs     ``asca`` (ANOVA-SCA), ``mixed_model`` (MixedLM), ``meba`` (time-series Hotelling)
Biomarker discovery      ``roc_feature``, ``biomarker_panel`` (nested CV)
Correlation              ``dgca`` (differential), ``corr_network`` (static within-condition)
Multi-omics integration  ``run_mofa`` (bridge to mofapy2 for metabol + RNA joint factors)
Multivariate             ``plsda``, ``opls_da`` (with VIP scores + Q²)
Pathway enrichment       ``msea_ora``, ``msea_gsea``, ``lion_enrichment``
Lipidomics (lipidr)      ``read_skyline``, ``summarize_transitions``,
                         ``normalize_pqn`` / ``normalize_istd``, ``de_lipids``,
                         ``lsea``, ``lipid_mva`` (pylipidr bridge)
Mass-based annotation    ``annotate_peaks``, ``mummichog_basic``
ID mapping               ``map_ids``
Database fetchers        ``fetch_kegg_pathways``, ``fetch_chebi_compounds``,
                         ``fetch_lion_associations``, ``fetch_hmdb_from_name``
Plotting                 ``volcano``, ``s_plot``, ``vip_bar``,
                         ``pathway_bar``, ``pathway_dot``

All scipy / sklearn / statsmodels / matplotlib imports are deferred
via module-level ``__getattr__`` — ``import omicverse.metabol`` itself
stays lightweight, each symbol is loaded only when first accessed.
Follows the same lazy-loading pattern used by the top-level
``omicverse`` package.
"""
from __future__ import annotations

import importlib as _importlib


# ---------------------------------------------------------------------------
# Lazy-attribute map: public symbol → (submodule path, attribute in submodule)
# ---------------------------------------------------------------------------
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # I/O — small module, but defer anyway for consistency
    "read_metaboanalyst":     (".io", "read_metaboanalyst"),
    "read_wide":              (".io", "read_wide"),
    "read_lcms":              (".io", "read_lcms"),
    # QC (pulls statsmodels for drift_correct)
    "cv_filter":              ("._qc", "cv_filter"),
    "drift_correct":          ("._qc", "drift_correct"),
    "blank_filter":           ("._qc", "blank_filter"),
    "sample_qc":              ("._qc", "sample_qc"),
    # Batch / drift correction (sklearn RF)
    "serrf":                  ("._batch", "serrf"),
    # Preprocessing
    "impute":                 ("._impute", "impute"),
    "normalize":              ("._norm", "normalize"),
    "transform":              ("._transform", "transform"),
    # Univariate stats (scipy.stats)
    "differential":           ("._stats", "differential"),
    "anova":                  ("._stats", "anova"),
    # Multi-factor designs (statsmodels.MixedLM + numpy SVD)
    "asca":                   ("._multifactor", "asca"),
    "ASCAEffect":             ("._multifactor", "ASCAEffect"),
    "ASCAResult":             ("._multifactor", "ASCAResult"),
    "mixed_model":            ("._multifactor", "mixed_model"),
    "meba":                   ("._multifactor", "meba"),
    # Biomarker discovery (sklearn)
    "roc_feature":            ("._biomarker", "roc_feature"),
    "biomarker_panel":        ("._biomarker", "biomarker_panel"),
    "BiomarkerPanelResult":   ("._biomarker", "BiomarkerPanelResult"),
    # Differential correlation (vectorised numpy + scipy)
    "dgca":                   ("._correlation", "dgca"),
    "corr_network":           ("._correlation", "corr_network"),
    # Multi-omics integration (bridge to ov.single.pyMOFA / mofapy2)
    "run_mofa":               ("._integration", "run_mofa"),
    # Multivariate (sklearn)
    "plsda":                  ("._plsda", "plsda"),
    "opls_da":                ("._plsda", "opls_da"),
    "PLSDAResult":            ("._plsda", "PLSDAResult"),
    # Pathway enrichment (scipy.stats + NumPy pre-ranked GSEA)
    "msea_ora":               ("._msea", "msea_ora"),
    "msea_gsea":              ("._msea", "msea_gsea"),
    "load_pathways":          ("._msea", "load_pathways"),
    # Mummichog (scipy.stats + heavy DataFrame work)
    "annotate_peaks":         ("._mummichog", "annotate_peaks"),
    "mummichog_basic":        ("._mummichog", "mummichog_basic"),
    "mummichog_external":     ("._mummichog", "mummichog_external"),
    # ID mapping
    "map_ids":                ("._id_mapping", "map_ids"),
    "normalize_name":         ("._id_mapping", "normalize_name"),
    # Database fetchers (network + urllib + gzip)
    "fetch_kegg_pathways":    ("._fetchers", "fetch_kegg_pathways"),
    "fetch_lion_associations": ("._fetchers", "fetch_lion_associations"),
    "fetch_chebi_compounds":  ("._fetchers", "fetch_chebi_compounds"),
    "fetch_hmdb_from_name":   ("._fetchers", "fetch_hmdb_from_name"),
    "clear_cache":            ("._fetchers", "clear_cache"),
    # Lipidomics (re + regex + scipy)
    "LipidIdentity":          ("._lipidomics", "LipidIdentity"),
    "parse_lipid":            ("._lipidomics", "parse_lipid"),
    "annotate_lipids":        ("._lipidomics", "annotate_lipids"),
    "aggregate_by_class":     ("._lipidomics", "aggregate_by_class"),
    "lion_enrichment":        ("._lipidomics", "lion_enrichment"),
    # Lipidomics — pylipidr bridge (Bioconductor lipidr workflow)
    "read_skyline":           ("._lipidomics", "read_skyline"),
    "add_sample_annotation":  ("._lipidomics", "add_sample_annotation"),
    "summarize_transitions":  ("._lipidomics", "summarize_transitions"),
    "normalize_pqn":          ("._lipidomics", "normalize_pqn"),
    "normalize_istd":         ("._lipidomics", "normalize_istd"),
    "de_lipids":              ("._lipidomics", "de_lipids"),
    "lsea":                   ("._lipidomics", "lsea"),
    "lipid_mva":              ("._lipidomics", "lipid_mva"),
    # Plotting (matplotlib)
    "volcano":                (".plotting", "volcano"),
    "s_plot":                 (".plotting", "s_plot"),
    "vip_bar":                (".plotting", "vip_bar"),
    "pathway_bar":            (".plotting", "pathway_bar"),
    "pathway_dot":            (".plotting", "pathway_dot"),
    "sample_qc_plot":         (".plotting", "sample_qc_plot"),
    "dgca_class_bar":         (".plotting", "dgca_class_bar"),
    "corr_network_plot":      (".plotting", "corr_network_plot"),
    "asca_variance_bar":      (".plotting", "asca_variance_bar"),
    "acyl_chain_map":         (".plotting", "acyl_chain_map"),
    # Lifecycle class
    "pyMetabo":               (".pymetabo", "pyMetabo"),
}

# Whole-submodule lazy loads — ``ov.metabol.plotting`` returns the module
_LAZY_SUBMODULES = {"plotting"}


# Submodules that host @register_function-decorated public API. Hydration
# imports every entry here so ``ov.export_registry()`` / ``ov.find_function``
# see metabol without waiting for a user to touch each function first.
_REGISTRY_SUBMODULES = (
    ".io",
    "._qc",
    "._batch",
    "._impute",
    "._norm",
    "._transform",
    "._stats",
    "._multifactor",
    "._biomarker",
    "._correlation",
    "._integration",
    "._plsda",
    "._msea",
    "._mummichog",
    "._id_mapping",
    "._fetchers",
    "._lipidomics",
    ".plotting",
)


def _hydrate_registry() -> None:
    """Import every decorator-bearing submodule to populate the global
    registry. Called from ``omicverse._registry._hydrate_registry_for_export``
    because metabol's lazy ``__init__`` would otherwise leave decorators
    un-executed at registry-export time."""
    for mod in _REGISTRY_SUBMODULES:
        try:
            _importlib.import_module(mod, __name__)
        except Exception:
            # Optional backends (gseapy, statsmodels) may be missing —
            # register whatever loads cleanly.
            continue


def __getattr__(name: str):
    """Module-level lazy import. Triggered on first access to every public
    symbol, so ``import omicverse.metabol`` itself does no heavy work."""
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value           # cache so subsequent access is free
        return value
    if name in _LAZY_SUBMODULES:
        module = _importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Make tab-completion and ``dir(ov.metabol)`` show the full lazy API."""
    return sorted(set(list(globals().keys()) + list(_LAZY_ATTRS.keys())
                      + list(_LAZY_SUBMODULES)))


__version__ = "0.4.0"

__all__ = [
    # class API
    "pyMetabo",
    # I/O
    "read_metaboanalyst",
    "read_wide",
    "read_lcms",
    # QC
    "cv_filter",
    "drift_correct",
    "blank_filter",
    "sample_qc",
    # batch / drift correction
    "serrf",
    # preprocessing
    "impute",
    "normalize",
    "transform",
    # stats
    "differential",
    "anova",
    # multi-factor designs
    "asca",
    "ASCAEffect",
    "ASCAResult",
    "mixed_model",
    "meba",
    # biomarker discovery
    "roc_feature",
    "biomarker_panel",
    "BiomarkerPanelResult",
    # differential correlation
    "dgca",
    "corr_network",
    # multi-omics integration
    "run_mofa",
    # multivariate
    "plsda",
    "opls_da",
    "PLSDAResult",
    # pathway enrichment
    "msea_ora",
    "msea_gsea",
    "load_pathways",
    "annotate_peaks",
    "mummichog_basic",
    "mummichog_external",
    # ID mapping
    "map_ids",
    "normalize_name",
    # database fetchers
    "fetch_kegg_pathways",
    "fetch_lion_associations",
    "fetch_chebi_compounds",
    "fetch_hmdb_from_name",
    "clear_cache",
    # lipidomics
    "LipidIdentity",
    "parse_lipid",
    "annotate_lipids",
    "aggregate_by_class",
    "lion_enrichment",
    # lipidomics — pylipidr bridge (Bioconductor lipidr workflow)
    "read_skyline",
    "add_sample_annotation",
    "summarize_transitions",
    "normalize_pqn",
    "normalize_istd",
    "de_lipids",
    "lsea",
    "lipid_mva",
    # plotting
    "plotting",
    "volcano",
    "s_plot",
    "vip_bar",
    "pathway_bar",
    "pathway_dot",
    "sample_qc_plot",
    "dgca_class_bar",
    "corr_network_plot",
    "asca_variance_bar",
    "acyl_chain_map",
]
