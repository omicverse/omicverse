r"""
Single-extracellular-vesicle (single-EV) proteomics — ``ov.single.ev``.

Analyses the protein content of *individual* extracellular vesicles
(exosomes / EVs). Single-EV proteomic data is an **EV x protein matrix** —
each row an individual vesicle, each column a protein / surface-marker
target — structurally analogous to a single-cell cell x gene matrix
(vesicle ~ cell, protein marker ~ gene). It is held in an
:class:`~anndata.AnnData` with ``X`` = EV x protein, ``obs`` = per-EV
metadata, ``var`` = protein targets, and ``uns['ev']`` carrying the
measurement ``value_type`` (``'count'`` / ``'intensity'`` / ``'binary'``).

Three measurement modalities are supported behind one API:

* **sequencing counts** — PBA (Proximity Barcoding Assay), DBS-Pro.
* **flow / imaging intensity** — NanoFCM nano-flow cytometry, ExoView /
  SP-IRIS, FCS files.
* **digital binary** — droplet digital ELISA / Simoa.

Generic preprocessing (scale / PCA / neighbors), graph clustering /
embedding (leiden / UMAP) and the standard single-cell plots are NOT
re-implemented here — use the omicverse-native ``ov.pp.*`` and ``ov.pl.*``
functions for those. Only the genuinely EV-specific steps live in this
module.

Pipeline stages
---------------
I/O                   ``read_ev_matrix``, ``read_pba``, ``read_exoview``,
                      ``read_nanofcm``, ``read_fcs``, ``read_digital_ev``,
                      ``simulate_ev``, ``refresh_ev_metrics``
QC                    ``qc``, ``subtract_isotype``, ``contaminant_score``,
                      ``detect_doublets``
Preprocessing         ``normalize`` (then ``ov.pp.scale`` / ``ov.pp.pca`` /
                      ``ov.pp.neighbors``)
Clustering            ``flowsom``, ``subpopulation_abundance`` (graph
                      clustering / UMAP via ``ov.pp.leiden`` / ``ov.pp.umap``)
Annotation            ``misev_markers``, ``classify_markers``,
                      ``annotate_ev_subtype``, ``tissue_of_origin``,
                      ``marker_enrichment``, ``purity_report``
Colocalization        ``colocalization``, ``coexpression_network``,
                      ``protein_combinations``, ``colocalization_plot``
Differential          ``rank_markers``, ``differential_abundance``,
                      ``differential_subpopulation``
Pseudo-bulk           ``pseudobulk``, ``pseudobulk_de``
Reporting             ``misev_report``, ``ev_summary``
Plotting              ``misev_marker_plot`` (embeddings / dotplots /
                      heatmaps / composition bars via ``ov.pl.*``)

Quick-start
-----------
>>> import omicverse as ov
>>> adata = ov.single.ev.read_pba('pba_counts.tsv')
>>> ov.single.ev.qc(adata)
>>> ov.single.ev.normalize(adata)
>>> ov.pp.scale(adata); ov.pp.pca(adata, layer='scaled')
>>> ov.pp.neighbors(adata, use_rep='scaled|original|X_pca')
>>> ov.single.ev.flowsom(adata, n_clusters=8)
>>> ov.single.ev.classify_markers(adata)
>>> ov.single.ev.annotate_ev_subtype(adata)
>>> ov.single.ev.colocalization(adata)
"""
from __future__ import annotations

import importlib as _importlib


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # --- I/O ---
    "read_ev_matrix":              (".io", "read_ev_matrix"),
    "read_pba":                    (".io", "read_pba"),
    "read_exoview":                (".io", "read_exoview"),
    "read_nanofcm":                (".io", "read_nanofcm"),
    "read_fcs":                    (".io", "read_fcs"),
    "read_digital_ev":             (".io", "read_digital_ev"),
    "simulate_ev":                 (".io", "simulate_ev"),
    "refresh_ev_metrics":          (".io", "refresh_ev_metrics"),
    # --- QC ---
    "qc":                          ("._qc", "qc"),
    "subtract_isotype":            ("._qc", "subtract_isotype"),
    "contaminant_score":           ("._qc", "contaminant_score"),
    "detect_doublets":             ("._qc", "detect_doublets"),
    # --- preprocessing ---
    "normalize":                   ("._preprocess", "normalize"),
    # --- clustering ---
    "flowsom":                     ("._cluster", "flowsom"),
    "subpopulation_abundance":     ("._cluster", "subpopulation_abundance"),
    # --- annotation ---
    "misev_markers":               ("._annotate", "misev_markers"),
    "classify_markers":            ("._annotate", "classify_markers"),
    "annotate_ev_subtype":         ("._annotate", "annotate_ev_subtype"),
    "tissue_of_origin":            ("._annotate", "tissue_of_origin"),
    "marker_enrichment":           ("._annotate", "marker_enrichment"),
    "purity_report":               ("._annotate", "purity_report"),
    # --- colocalization ---
    "colocalization":              ("._coloc", "colocalization"),
    "coexpression_network":        ("._coloc", "coexpression_network"),
    "protein_combinations":        ("._coloc", "protein_combinations"),
    "colocalization_plot":         ("._coloc", "colocalization_plot"),
    # --- differential ---
    "rank_markers":                ("._markers", "rank_markers"),
    "differential_abundance":      ("._markers", "differential_abundance"),
    "differential_subpopulation":  ("._markers", "differential_subpopulation"),
    # --- pseudo-bulk ---
    "pseudobulk":                  ("._pseudobulk", "pseudobulk"),
    "pseudobulk_de":               ("._pseudobulk", "pseudobulk_de"),
    # --- reporting ---
    "misev_report":                ("._report", "misev_report"),
    "ev_summary":                  ("._report", "ev_summary"),
    # --- plotting ---
    "misev_marker_plot":           (".plotting", "misev_marker_plot"),
}

_REGISTRY_SUBMODULES = (
    ".io",
    "._qc",
    "._preprocess",
    "._cluster",
    "._annotate",
    "._coloc",
    "._markers",
    "._pseudobulk",
    "._report",
    ".plotting",
)


def _hydrate_registry() -> None:
    """Force-import every @register_function-bearing submodule so the global
    registry sees ``ov.single.ev`` at export time."""
    for mod in _REGISTRY_SUBMODULES:
        try:
            _importlib.import_module(mod, __name__)
        except Exception:
            continue


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys()) + list(_LAZY_ATTRS.keys())))


__version__ = "0.1.0"

__all__ = list(_LAZY_ATTRS.keys())
