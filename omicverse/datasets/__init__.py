r"""
Dataset management utilities for omicverse.

This module provides functions to load datasets for single-cell analysis,
following the dynamo-release sample_data.py pattern.

Main functions:
    get_adata: Download and load AnnData from URLs
    download_data: Download files with progress tracking
    pbmc3k: Load PBMC 3k dataset with fallback to mock data
    create_mock_dataset: Generate synthetic datasets for testing
    
Dataset functions:
    - Scanpy-inspired: blobs, burczynski06, moignard15, paul15, pbmc68k_reduced
    - Simulations: toggleswitch, krumsiek11
    - Dynamo datasets: scnt_seq_neuron_splicing, scnt_seq_neuron_labeling
    - Real datasets: zebrafish, dentate_gyrus, bone_marrow, hematopoiesis
    - Special: multi_brain_5k (multiome data)
    
Core features:
    - Robust download with error handling and retry logic
    - Progress tracking with tqdm
    - Support for h5ad and loom formats
    - Mock data generation for testing
    
Examples:
    >>> import omicverse as ov
    >>> 
    >>> # Load PBMC 3k data (with clustering)
    >>> adata = ov.datasets.pbmc3k(processed=True)
    >>> print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    >>> 
    >>> # Load specific datasets
    >>> adata = ov.datasets.hematopoiesis()
    >>> adata = ov.datasets.paul15()  # Myeloid development
    >>> adata = ov.datasets.blobs()   # Synthetic clusters
    >>> 
    >>> # Create mock data
    >>> adata = ov.datasets.create_mock_dataset(
    ...     n_cells=1000, 
    ...     n_cell_types=5,
    ...     with_clustering=True
    ... )
"""

from ._datasets import (
    # Core utilities
    download_data,
    download_data_requests,
    get_adata,
    pancreas_cellrank,
    
    # Main dataset loaders
    pbmc3k,
    bhattacherjee,
    create_mock_dataset,
    
    # Scanpy-inspired datasets
    blobs,
    burczynski06,
    moignard15,
    paul15,
    toggleswitch,
    krumsiek11,
    
    # Placeholder functions
    gillespie,
    hl60,
    nascseq,
    scslamseq,
    scifate,
    cite_seq,
    
    # Real dataset functions
    scnt_seq_neuron_splicing,
    scnt_seq_neuron_labeling,
    zebrafish,
    dentate_gyrus,
    bone_marrow,
    haber,
    hg_forebrain_glutamatergic,
    chromaffin,
    bm,
    pancreatic_endocrinogenesis,
    dentate_gyrus_scvelo,
    sceu_seq_rpe1,
    sceu_seq_organoid,
    hematopoiesis,
    hematopoiesis_raw,
    human_tfs,
    multi_brain_5k,

    decov_bulk_covid_bulk,
    decov_bulk_covid_single,

    sc_ref_Lymph_Node,
    visium_lymph_node,
    pbmc8k,
    seqfish,
)
from ._protein import (
    # Real proteomics datasets for the ov.protein tutorials
    protein_pxd000022,
    protein_pxd000279,
    protein_pxd000438,
    protein_dda_spikein,
    protein_dia,
    protein_olink,
)
from ._genetics import (
    # Real genetics datasets for the ov.genetics tutorials
    geuvadis_genotype,
    geuvadis_expression,
    gwas_sumstats,
    gtex_eqtl,
    genetics_scrna,
    recombination_map,
    gene_annotation,
)
from ._timecourse import (
    # Real bulk RNA-seq time-course datasets for the ov.bulk tutorials
    fission_timecourse,
    pombe_genesets,
)
from ._airr import (
    # Real immune-repertoire (AIRR) datasets for the ov.airr tutorials
    airr_singlecell,
    airr_singlecell_bcr,
    airr_bcr,
    airr_tcr_antigen,
    vdjdb_reference,
)
from ._ev import (
    # Real single-extracellular-vesicle (single-EV) proteomic datasets
    # for the ov.single.ev tutorials
    ev_pba,
    ev_masev,
    ev_marker_reference,
)
from ._flow import (
    # SIMULATED flow-cytometry sample for the ov.flow tutorials. The only
    # generated dataset in this namespace — a real FCS file cannot teach
    # compensation, because the uncompensated truth is not recoverable from it.
    flow_demo,
    flow_demo_fcs,
    FLOW_DEMO_PANEL,
    FLOW_DEMO_SPILLOVER,
)
from ._cytometry import (
    # Real public flow-cytometry datasets for the ov.flow tutorials, fetched
    # from their original repositories (Zenodo / PLOS) rather than re-hosted
    flow_pbmc_fortessa,
    flow_pbmc_spectral,
)
from ._ambient import (
    # Real raw 10x droplet datasets for the ov.pp.ambient tutorial
    pbmc_raw_10x,
    hgmm_mixture,
)
from ._metabolism import (
    # Real scRNA-seq + precomputed Compass output for the ov.single
    # metabolism tutorial (Metabolism / MetaboliteCCC)
    metabolism_hnsc,
    metabolism_compass,
)
from ._spatial import (
    # SPATA2's own example Visium slide, for the ov.space utility tutorial
    spata2_example,
)
from ._signatures import load_signatures_from_file, predefined_signatures
from ._crossspecies import saturn_frog_zebrafish
