r"""
Utility functions for data handling and analysis.

This module provides essential utility functions for data I/O, manipulation,
visualization, clustering, and various computational tasks across the omicverse
ecosystem. It serves as the foundation for many higher-level analysis functions.

Data I/O and manipulation:
    read: Universal data reader for various formats
    read_csv, read_10x_mtx, read_h5ad, read_10x_h5: Format-specific readers
    data_downloader: Download datasets and models
    geneset_prepare: Prepare gene sets for analysis
    get_gene_annotation: Gene annotation utilities
    
Visualization utilities:
    palette: Color palette management
    ov_plot_set, plot_set: Global plot settings
    Various plotting helper functions
    
Clustering and analysis:
    cluster: General clustering functions
    LDA_topic: Topic modeling for single-cell data
    filtered, refine_label: Data filtering and label refinement
    
Statistical and computational tools:
    correlation_pseudotime: Pseudotime correlation analysis
    np_mean, np_std: Numpy-based statistical functions
    neighbors: Neighbor graph construction
    mde: Minimum distortion embedding
    
Data structures and conversion:
    anndata_sparse: Sparse matrix utilities for AnnData
    store_layers, retrieve_layers: Layer management
    
Specialized analysis:
    roe: Ratio of expression analysis
    cal_paga, plot_paga: PAGA trajectory analysis
    venny4py: Venn diagram utilities
    syn: Synthetic data generation
    scatterplot: Advanced scatter plot functions
    knn: K-nearest neighbors utilities
    heatmap: Heatmap generation and customization
    lsi: Latent semantic indexing

Examples:
    >>> import omicverse as ov
    >>> # Data reading
    >>> adata = ov.utils.read('data.h5ad')
    >>> 
    >>> # Clustering
    >>> labels = ov.utils.cluster(
    ...     adata.obsm['X_pca'], 
    ...     method='leiden', 
    ...     resolution=0.5
    ... )
    >>> 
    >>> # Visualization setup
    >>> ov.utils.ov_plot_set()
    >>> 
    >>> # Download data
    >>> ov.utils.data_downloader('pbmc3k')
"""

# All functions imported via wildcard imports from submodules
from ._data import (
    read,read_csv,read_10x_mtx,read_h5ad,read_10x_h5,convert_to_pandas,wrap_dataframe,
    download_CaDRReS_model,download_GDSC_data,
    download_pathway_database,download_geneid_annotation_pair,
    gtf_to_pair_tsv,download_tosica_gmt,geneset_prepare,get_gene_annotation,
    correlation_pseudotime,store_layers,retrieve_layers,easter_egg,
    save,load,convert_adata_for_rust,
    anndata_sparse,np_mean,np_std,
    load_signatures_from_file,predefined_signatures,print_tree
    )
from ._anndata_rust_patch import patch_rust_adata
from ._plot import (
    plot_set,plotset,ov_plot_set,pyomic_palette,palette,blue_palette,orange_palette,
    red_palette,green_palette,plot_text_set,ticks_range,plot_boxplot,plot_network,
    plot_cellproportion,plot_embedding_celltype,geneset_wordcloud,
    plot_pca_variance_ratio,gen_mpl_labels
)
#from ._genomics import *
from ._mde import mde
from ._syn import logger, pancreas, synthetic_iid, url_datadir
from ._scatterplot import diffmap, draw_graph, embedding, pca, spatial, tsne, umap
from ._knn import weighted_knn_trainer, weighted_knn_transfer
from ._heatmap import (
    additional_colors,
    adjust_palette,
    clip,
    default_color,
    default_palette,
    get_colors,
    interpret_colorkey,
    is_categorical,
    is_list,
    is_list_of_str,
    is_list_or_array,
    is_view,
    make_dense,
    plot_heatmap,
    set_colors_for_categorical_obs,
    strings_to_categoricals,
    to_list,
)
from ._roe import roe, roe_plot_heatmap, transform_roe_values
from ._odds_ratio import odds_ratio, plot_odds_ratio_heatmap
from ._shannon_diversity import shannon_diversity, compare_shannon_diversity, plot_shannon_diversity
from ._resolution import optimal_resolution, plot_resolution_optimization, resolution_stability_analysis
from ._paga import cal_paga,plot_paga,PAGA_tree
from ._cluster import cluster,LDA_topic,filtered,refine_label
from ._venn import venny4py
from ._lsi import Array, lsi, tfidf
from ._neighboors import neighbors,calc_kBET,calc_kSIM
from ._gene_id_conversion import _infer_species_and_release, convert2gene_symbol, convert2symbol, id2symbol, convert2gene_id, symbol2id
from ._sample_metadata_alignment import (
    PreflightResult,
    preflight_alignment,
    align_to_common,
    align_samples,
)
from ._metabolights import load_metabolights
from ._ovagent_lookup import (
    RegistryScanner,
    initialize_skill_registry,
    registry_lookup,
    registry_summary,
    skill_lookup,
)

# Import smart_agent module to make it accessible and expose key entrypoints
# Store verifier with a private name first to ensure reference is preserved
from . import agent_backend, agent_backend_streaming, biocontext, smart_agent
from . import ovagent  # ensure ovagent subpackage is a direct attribute
from . import verifier as _verifier_module
from .agent_backend import BackendConfig, OmicVerseLLMBackend, Usage
from .smart_agent import Agent, OmicVerseAgent, list_supported_models

# P0-2 / P0-3 / P1-1 / P2-1 / P2-2: New agent infrastructure modules
from .agent_config import AgentConfig, HarnessConfig, SandboxFallbackPolicy
from .agent_errors import (
    OVAgentError, WorkflowNeedsFallback, ProviderError,
    ConfigError, ExecutionError, SandboxDeniedError,
)
from .agent_reporter import AgentEvent, EventLevel, Reporter, make_reporter
from .context_compactor import ContextCompactor, estimate_tokens
from .session_history import SessionHistory, HistoryEntry
from . import gpuex  # noqa: F401  — ov.utils.gpuex.scipy.rankdata etc.
from .harness import (
    HARNESS_EVENT_TYPES,
    STREAM_EVENT_TYPES,
    ArtifactRef,
    HarnessEvent,
    RunTrace,
    RunTraceRecorder,
    RunTraceStore,
    StepTrace,
    build_stream_event,
    make_step_id,
    make_trace_id,
    make_turn_id,
)

# Python 3.10 compatibility: Provide __getattr__ to dynamically return verifier
# This ensures getattr(omicverse.utils, 'verifier') works in unittest.mock.patch
def __getattr__(name):
    """Dynamically return module attributes for submodule and symbol resolution.

    Handles two cases that arise when sys.modules is manipulated (e.g. by test
    isolation stubs) and Python's normal import-time attribute-setting on the
    parent package is skipped:

    1. Submodule lookup: if ``omicverse.utils.<name>`` is in ``sys.modules``
       but was never set as an attribute on *this* module object (common when a
       test replaces ``sys.modules['omicverse.utils']`` with a fresh stub),
       return and cache the submodule so that ``getattr(omicverse.utils, name)``
       and ``monkeypatch.setattr("omicverse.utils.<name>.<sym>", ...)`` work.
    2. Legacy symbol forwarding for ``verifier`` and smart-agent entrypoints.
    """
    import sys as _sys

    # Fast-path: check if a submodule with this name is already imported
    fqn = f"{__name__}.{name}"
    submod = _sys.modules.get(fqn)
    if submod is not None:
        # Cache it on the module so future lookups are O(1)
        globals()[name] = submod
        return submod

    if name == 'verifier':
        return _verifier_module
    if name in {'Agent', 'OmicVerseAgent', 'list_supported_models'}:
        return getattr(smart_agent, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Also make verifier accessible via normal attribute access
verifier = _verifier_module

# Explicit public exports for stable, non-wildcard imports
__all__ = [
    # @ _metabolights
    "load_metabolights",
    # @ _data
    "read",
    "read_csv",
    "read_10x_mtx",
    "read_h5ad",
    "read_10x_h5",
    "convert_to_pandas",
    "wrap_dataframe",
    "download_CaDRReS_model",
    "download_GDSC_data",
    "download_pathway_database",
    "download_geneid_annotation_pair",
    "gtf_to_pair_tsv",
    "download_tosica_gmt",
    "geneset_prepare",
    "get_gene_annotation",
    "correlation_pseudotime",
    "store_layers",
    "retrieve_layers",
    "easter_egg",
    "save",
    "load",
    "convert_adata_for_rust",
    "anndata_sparse",
    "np_mean",
    "np_std",
    "load_signatures_from_file",
    "predefined_signatures",
    "print_tree",
    # @ _anndata_rust_patch
    "patch_rust_adata",
    # @ _plot
    "plot_set",
    "plotset",
    "ov_plot_set",
    "pyomic_palette",
    "palette",
    "blue_palette",
    "orange_palette",
    "red_palette",
    "green_palette",
    "plot_text_set",
    "ticks_range",
    "plot_boxplot",
    "plot_network",
    "plot_cellproportion",
    "plot_embedding_celltype",
    "geneset_wordcloud",
    "plot_pca_variance_ratio",
    "gen_mpl_labels",
    # @ _mde
    "mde",
    # @ _syn
    "logger",
    "pancreas",
    "synthetic_iid",
    "url_datadir",
    # @ _scatterplot
    "diffmap",
    "draw_graph",
    "embedding",
    "pca",
    "spatial",
    "tsne",
    "umap",
    # @ _knn
    "weighted_knn_trainer",
    "weighted_knn_transfer",
    # @ _heatmap
    "additional_colors",
    "adjust_palette",
    "clip",
    "default_color",
    "default_palette",
    "get_colors",
    "interpret_colorkey",
    "is_categorical",
    "is_list",
    "is_list_of_str",
    "is_list_or_array",
    "is_view",
    "make_dense",
    "plot_heatmap",
    "set_colors_for_categorical_obs",
    "strings_to_categoricals",
    "to_list",
    # @ _roe
    "roe",
    "roe_plot_heatmap",
    "transform_roe_values",
    # @ _odds_ratio
    "odds_ratio",
    "plot_odds_ratio_heatmap",
    # @ _shannon_diversity
    "shannon_diversity",
    "compare_shannon_diversity",
    "plot_shannon_diversity",
    # @ _resolution
    "optimal_resolution",
    "plot_resolution_optimization",
    "resolution_stability_analysis",
    # @ _paga
    "cal_paga",
    "plot_paga",
    "PAGA_tree",
    # @ _cluster
    "cluster",
    "LDA_topic",
    "filtered",
    "refine_label",
    # @ _venn
    "venny4py",
    # @ _lsi
    "Array",
    "lsi",
    "tfidf",
    # @ _neighboors
    "neighbors",
    "calc_kBET",
    "calc_kSIM",
    # @ _gene_id_conversion
    "convert2gene_symbol",
    "convert2symbol",
    "id2symbol",
    "convert2gene_id",
    "symbol2id",
    # @ _ovagent_lookup
    "RegistryScanner",
    "initialize_skill_registry",
    "registry_lookup",
    "registry_summary",
    "skill_lookup",
    # @ agent_backend
    "agent_backend",
    "agent_backend_streaming",
    "biocontext",
    "BackendConfig",
    "OmicVerseLLMBackend",
    "Usage",
    # @ ovagent
    "ovagent",
    # @ smart_agent
    "smart_agent",
    "Agent",
    "OmicVerseAgent",
    "list_supported_models",
    "HarnessConfig",
    # @ harness
    "HARNESS_EVENT_TYPES",
    "STREAM_EVENT_TYPES",
    "ArtifactRef",
    "HarnessEvent",
    "RunTrace",
    "RunTraceRecorder",
    "RunTraceStore",
    "StepTrace",
    "build_stream_event",
    "make_step_id",
    "make_trace_id",
    "make_turn_id",
    # @ verifier
    "verifier",
    # @ agent_config
    "AgentConfig",
    "SandboxFallbackPolicy",
    # @ agent_errors — public compatibility shims and error hierarchy
    "OVAgentError",
    "WorkflowNeedsFallback",
    "ProviderError",
    "ConfigError",
    "ExecutionError",
    "SandboxDeniedError",
    # @ agent_reporter
    "AgentEvent",
    "EventLevel",
    "Reporter",
    "make_reporter",
    # @ context_compactor
    "ContextCompactor",
    "estimate_tokens",
    # @ session_history
    "SessionHistory",
    "HistoryEntry",
    # @ _gene_id_conversion (private, re-exported for compatibility)
    "_infer_species_and_release",
]
