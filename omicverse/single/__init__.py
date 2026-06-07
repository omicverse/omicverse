r"""
Single-cell omics analysis utilities.

This module provides comprehensive tools for single-cell RNA-seq analysis including:
- Cell type annotation and automated identification
- Trajectory inference and pseudotime analysis
- Gene regulatory network analysis
- Cell-cell communication analysis
- Multi-modal integration (RNA + ATAC)
- Drug response prediction
- Pathway enrichment and functional analysis

Key classes:
    pyMOFA: Multi-Omics Factor Analysis
    StaVIA: Spatial/temporal trajectory inference with the VIA backend
    pyVIA: Velocity and pseudotime analysis 
    pySIMBA: Single-cell integration and batch alignment
    pyTOSICA: Trajectory inference and cell fate analysis
    pyCEFCON: Cell-cell communication analysis
    MetaCell: Metacell construction and analysis
    pyDEG: Single-cell differential expression

Annotation tools:
    pySCSA: Automated cell type annotation
    MetaTiME: Tumor microenvironment annotation
    gptcelltype: GPT-based cell type identification
    CellVote: Consensus cell type annotation

Trajectory analysis:
    StaVIA: Spatial/temporal trajectory inference
    TrajInfer: Trajectory inference framework
    scLTNN: Lineage tracing with neural networks
    cytotrace2: Developmental potential scoring

Examples:
    >>> import omicverse as ov
    >>> # Automated cell annotation
    >>> ov.single.pySCSA(adata, tissue='lung', species='human')
    >>> 
    >>> # Trajectory analysis
    >>> stavia = ov.single.StaVIA(adata, spatial_key='spatial')
    >>> stavia.fit()
    >>> 
    >>> # Multi-omics integration
    >>> mofa = ov.single.pyMOFA(data_dict)
    >>> mofa.build_mofa()
"""
import importlib
from .._optional import bind_optional_symbols

# Heavy functionality lives in submodules and is imported lazily.

# Core gene selection and differential expression
from ._cosg import cosg
from ._anno import pySCSA,MetaTiME,scanpy_lazy,scanpy_cellanno_from_dict,get_celltype_marker
from ._mofa import (
    pyMOFAART,pyMOFA,GLUE_pair,
    factor_exact,factor_correlation,
    get_weights,glue_pair,get_r2_from_hdf5,
    convert_r2_to_matrix,factor_group_correlation_mdata,
    plot_factor_group_associations,plot_factor_boxplots,
    plot_factors_violin,plot_weights,
)
from ._scdrug import writeGEP, Drug_Response
from ._autoresolution import auto_resolution, autoResolution
from ._cpdb import (
    cpdb_network_cal,cpdb_plot_network,
    cpdb_plot_interaction,
    cpdb_interaction_filtered,
    cpdb_submeans_exacted,cpdb_exact_target,
    cpdb_exact_source,cellphonedb_v5,
    run_cellphonedb_v5, format_cpdb_results, format_cpdb_results_for_viz,
    download_cellphonedb_database, validate_cpdb_database,
)
from ._liana import run_liana, format_liana_results, format_liana_results_for_viz
from ._comm import to_comm_adata, extract_comm_adata

from ._scgsea import (
    geneset_aucell,pathway_aucell,pathway_aucell_enrichment,
    geneset_aucell_tmp,pathway_aucell_tmp,pathway_aucell_enrichment_tmp,
    pathway_enrichment,pathway_enrichment_plot
)
from ._via import pyVIA, hematopoiesis
scRNA_hematopoiesis = hematopoiesis
from ._atac import atac_concat_get_index,atac_concat_inner,atac_concat_outer
from ._batch import batch_correction
from ._diffusionmap import diffmap
from ._aucell import aucell
from ._metacell import (
    MetaCell,
    plot_metacells,
    get_obs_value,
    optimize_granularity,
    compare_metacell_backends,
)
from ._pseudobulk import pseudobulk
from ._mdic3 import pyMDIC3
from ._gptcelltype import gptcelltype,gpt4celltype,get_cluster_celltype
from ._gptcelltype_local import gptcelltype_local
from ._sccaf import SCCAF_assessment,plot_roc,SCCAF_optimize_all,color_long
from ._multimap import TFIDF_LSI,Wrapper,Integration,Batch
from ._cellvote import get_cluster_celltype,CellVote,cellvote_consensus_score
from ._deg_ct import DCT,DEG
from ._lazy_function import lazy
from ._lazy_report import generate_scRNA_report
from ._lazy_checkpoint import lazy_checkpoint, resume_from_checkpoint, list_checkpoints, cleanup_checkpoints

# Monocle2-style trajectory analysis (pure Python re-implementation).
# Usage: mono = ov.single.Monocle(adata); mono.preprocess(); ...
from ._monocle import Monocle

# Pseudotime → terminal-state → fate-probability unification.
# Usage: fate = ov.single.PseudotimeFate(adata, pseudotime_key='slingshot_pseudotime'); fate.fit()
# Works downstream of any TrajInfer backend (slingshot / palantir / scorpius / …).
from ._pseudotime_fate import PseudotimeFate, PseudotimeFateResult

# Single-cell CNV inference — pure-Python CopyKAT / inferCNV wrappers.
# Usage: cnv = ov.single.CNV(adata, method='copykat'); cnv.run()
from ._cnv import CNV

# Single-cell metabolism — scMetabolism / Compass / scFEA + MEBOCOST.
# Usage: met = ov.single.Metabolism(adata, method='scmetabolism'); met.run()
from ._metabolism import Metabolism, MetaboliteCCC, differential_metabolism

from ._stavia import StaVIA
from ._lazy_step_by_step import (
    lazy_step_qc, lazy_step_preprocess, lazy_step_scale, lazy_step_pca,
    lazy_step_cell_cycle, lazy_step_harmony, lazy_step_scvi, 
    lazy_step_select_best_method, lazy_step_mde, lazy_step_clustering,
    lazy_step_final_embeddings, lazy_step_by_step_guide
)
from ._diffusionmap import diffmap
from ._cellmatch import CellOntologyMapper,download_cl
from ._scenic import SCENIC, grn, build_correlation_network_umap_layout, add_tf_regulation, plot_grn
from ._annotation import Annotation
from ._annotation_ref import AnnotationRef
from ._velo import (
    Velo,
    velocity_embedding,
    velocity,
    cellrank_fate,
    state_names,
    clean_lineages,
    perturbation_effect,
    cell_fate_perturbation,
    velocity_effect,
)
from ._milo_dev import Milo
from ._augur import Augur
from ._markers import find_markers, get_markers
from ._perturb import PerturbResult, perturb, lineage_pseudotime
from ._dynamic_features import DynamicFeaturesResult, dynamic_features
from ._cefcon import (
    convert_human_to_mouse_network,
    load_human_prior_interaction_network,
    mouse_hsc_nestorowa16,
)

_TORCH_DEPS = ("torch", "torch_geometric")

bind_optional_symbols(
    globals(),
    "._nocd",
    ["scnocd"],
    package=__name__,
    feature="omicverse.single.scnocd",
    dependencies=_TORCH_DEPS,
)

bind_optional_symbols(
    globals(),
    "._simba",
    ["pySIMBA"],
    package=__name__,
    feature="omicverse.single.pySIMBA",
    dependencies=("torch",),
)

bind_optional_symbols(
    globals(),
    "._tosica",
    ["pyTOSICA"],
    package=__name__,
    feature="omicverse.single.pyTOSICA",
    dependencies=("torch",),
)

bind_optional_symbols(
    globals(),
    "._cellfategenie",
    ["Fate", "gene_trends", "mellon_density"],
    package=__name__,
    feature="omicverse.single.cell fate analysis",
    dependencies=("torch",),
)

bind_optional_symbols(
    globals(),
    "._ltnn",
    ["scLTNN", "plot_origin_tesmination", "find_related_gene"],
    package=__name__,
    feature="omicverse.single.scLTNN",
    dependencies=("torch",),
)

bind_optional_symbols(
    globals(),
    "._traj",
    ["TrajInfer", "fle"],
    package=__name__,
    feature="omicverse.single.TrajInfer",
    dependencies=("torch", "igraph"),
)

bind_optional_symbols(
    globals(),
    "._cefcon",
    ["pyCEFCON"],
    package=__name__,
    feature="omicverse.single.pyCEFCON",
    dependencies=_TORCH_DEPS,
)

bind_optional_symbols(
    globals(),
    "._cytotrace2",
    ["cytotrace2"],
    package=__name__,
    feature="omicverse.single.cytotrace2",
    dependencies=("torch",),
)

bind_optional_symbols(
    globals(),
    "._scdiffusion",
    ["scDiffusion"],
    package=__name__,
    feature="omicverse.single.scDiffusion",
    dependencies=("torch",),
)


def __getattr__(name):
    if name == "popv":
        return importlib.import_module(".popv", package=__name__)
    if name == "ev":
        return importlib.import_module(".ev", package=__name__)
    if name in {"cNMF", "Hotspot"}:
        from . import _cnmf as cnmf_module

        return getattr(cnmf_module, name)
    if name == "NMF":
        from . import _nmf as nmf_module

        globals()["NMF"] = nmf_module.NMF
        return nmf_module.NMF
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Core analysis functions
    'cosg',
    'lazy',
    'lazy_checkpoint', 
    'resume_from_checkpoint',
    'list_checkpoints',
    'cleanup_checkpoints',
    'dynamic_features',
    'DynamicFeaturesResult',
    'lazy_step_qc',
    'lazy_step_preprocess',
    'lazy_step_scale',
    'lazy_step_pca',
    'lazy_step_cell_cycle',
    'lazy_step_harmony',
    'lazy_step_scvi',
    'lazy_step_select_best_method',
    'lazy_step_mde',
    'lazy_step_clustering',
    'lazy_step_final_embeddings',
    'lazy_step_by_step_guide',
    'aucell',
    
    # Cell type annotation
    'pySCSA',
    'MetaTiME', 
    'scanpy_lazy',
    'scanpy_cellanno_from_dict',
    'get_celltype_marker',
    'gptcelltype',
    'gpt4celltype',
    'get_cluster_celltype',
    'gptcelltype_local',
    'CellVote',
    'CellOntologyMapper',
    'download_cl',
    'popv',
    
    # Multi-omics integration
    'pyMOFAART',
    'pyMOFA',
    'GLUE_pair',
    'factor_exact',
    'factor_correlation',
    'get_weights',
    'glue_pair',
    'get_r2_from_hdf5',
    'pySIMBA',
    'pyTOSICA',
    'TFIDF_LSI',
    'Wrapper',
    'Integration',
    'Batch',
    'convert_r2_to_matrix',
    'factor_group_correlation_mdata',
    'plot_factor_group_associations',
    'plot_factor_boxplots',
    'plot_factors_violin',
    'plot_weights',
    
    # Trajectory and pseudotime analysis
    'StaVIA',
    'pyVIA',
    'hematopoiesis',
    'scRNA_hematopoiesis',
    'TrajInfer',
    'PseudotimeFate',
    'PseudotimeFateResult',
    'fle',
    'diffmap',
    'scLTNN',
    'plot_origin_tesmination',
    'find_related_gene',
    'cytotrace2',
    
    # Cell fate and development
    'Fate',
    'gene_trends',
    'mellon_density',
    
    # Network and communication analysis
    'scnocd',
    'pyCEFCON',
    'velocity',
    'cellrank_fate',
    'state_names',
    'clean_lineages',
    'perturbation_effect',
    'cell_fate_perturbation',
    'velocity_effect',
    'convert_human_to_mouse_network',
    'load_human_prior_interaction_network',
    'mouse_hsc_nestorowa16',
    'cpdb_network_cal',
    'cpdb_plot_network',
    'cpdb_plot_interaction',
    'cpdb_interaction_filtered',
    'cpdb_submeans_exacted',
    'cpdb_exact_target',
    'cpdb_exact_source',
    'cellphonedb_v5',
    'run_cellphonedb_v5',
    'download_cellphonedb_database',
    'validate_cpdb_database',
    'format_cpdb_results',
    'format_cpdb_results_for_viz',
    'run_liana',
    'format_liana_results',
    'format_liana_results_for_viz',
    'to_comm_adata',
    'extract_comm_adata',
    # Pathway and functional analysis
    'geneset_aucell',
    'pathway_aucell',
    'pathway_aucell_enrichment',
    'geneset_aucell_tmp',
    'pathway_aucell_tmp',
    'pathway_aucell_enrichment_tmp',
    'pathway_enrichment',
    'pathway_enrichment_plot',
    
    # Drug response analysis
    'auto_resolution',
    'autoResolution',  # backward-compatible camelCase alias
    'writeGEP',
    'Drug_Response',
    'scDiffusion',
    
    # ATAC-seq analysis
    'atac_concat_get_index',
    'atac_concat_inner',
    'atac_concat_outer',
    
    # Batch correction and preprocessing  
    'batch_correction',
    
    # Quality control and assessment
    'SCCAF_assessment',
    'plot_roc',
    'SCCAF_optimize_all',
    'color_long',
    
    # Metacells and aggregation
    'MetaCell',
    'plot_metacells',
    'get_obs_value',
    'optimize_granularity',
    'compare_metacell_backends',
    'pseudobulk',

    # Single-cell CNV
    'CNV',

    # Single-cell metabolism
    'Metabolism',
    'MetaboliteCCC',
    'differential_metabolism',

    # Differential expression
    'DCT',
    'DEG',
    'pyMDIC3',
    
    # Additional analysis tools  
    'cnmf',
    'cNMF',
    'NMF',
    'Hotspot',
    'generate_scRNA_report',
    'SCENIC',
    'grn',
    'build_correlation_network_umap_layout',
    'add_tf_regulation',
    'plot_grn',
    'Velo',
    'velocity_embedding',
    'Annotation',   # cell type annotation
    'AnnotationRef', # cell type annotation with reference
    'Milo',
    'Augur',
    # Marker gene utilities
    'find_markers',
    'get_markers',
]
