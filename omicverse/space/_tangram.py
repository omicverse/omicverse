r"""Module providing encapsulation of Tangram for spatial deconvolution.

This module implements a wrapper for the Tangram algorithm, which enables mapping
between single-cell RNA sequencing data and spatial transcriptomics data. The main
functionality includes cell type deconvolution, spatial mapping, and gene expression
imputation.
"""
from typing import Any
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
from .._settings import add_reference
from .._registry import register_function

tg_install=False

@register_function(
    aliases=["Tangram空间解卷积", "Tangram", "spatial_deconvolution", "空间解卷积分析", "空间映射"],
    category="space",
    description="Tangram spatial deconvolution for mapping scRNA-seq cell types to spatial locations",
    prerequisites={
        'functions': []  # Requires scRNA-seq with annotations and spatial data
    },
    requires={
        'obs': []  # Requires clusters column in scRNA-seq (user-specified)
    },
    produces={
        'obsm': ['tangram_ct_pred']  # Cell type predictions in spatial data
    },
    auto_fix='escalate',
    examples=[
        "# Basic Tangram analysis",
        "tangram = ov.space.Tangram(adata_sc=sc_adata, adata_sp=spatial_adata,",
        "                          clusters='cell_type')",
        "# Train mapping model",
        "tangram.train(mode='clusters', num_epochs=500)",
        "# Project cell types to spatial locations",
        "adata_mapped = tangram.cell2location()",
        "# Gene imputation",
        "adata_imputed = tangram.impute()",
        "# Custom marker selection",
        "tangram = ov.space.Tangram(sc_adata, spatial_adata,",
        "                          clusters='leiden', marker_size=200)",
        "# Access mapping results",
        "mapping_matrix = tangram.ad_map"
    ],
    related=["space.clusters", "bulk.pyDEG", "space.svg"]
)
class Tangram(object):
    r"""Tangram spatial deconvolution class for cell type mapping.
    
    Tangram is a method for integrating single-cell RNA sequencing (scRNA-seq) data
    with spatial transcriptomics data. It enables:
    1. Mapping cell types from scRNA-seq to spatial locations
    2. Deconvolving cell type proportions in spatial spots
    3. Imputing gene expression in spatial data
    4. Analyzing spatial organization of cell types

    The method works by:
    1. Identifying marker genes for each cell type
    2. Training a mapping model using these markers
    3. Projecting cell type annotations to spatial coordinates
    4. Optionally imputing full gene expression profiles

    Parameters
    ----------
    adata_sc : AnnData
        Single-cell reference AnnData with cell-type labels.
    adata_sp : AnnData
        Spatial AnnData to receive projected cell-type information.
    clusters : str, default=''
        Cell-type column name in ``adata_sc.obs``.
    marker_size : int, default=100
        Number of marker genes selected per cell type for mapping.
    gene_to_lowercase : bool, default=False
        Whether to lowercase gene names before matching genes across datasets.

    Attributes:
        adata_sc: AnnData
            Single-cell RNA-seq data with:
            - Gene expression matrix in X
            - Cell type annotations in obs[clusters]
        adata_sp: AnnData
            Spatial transcriptomics data with:
            - Gene expression matrix in X
            - Spatial coordinates in obsm['spatial']
        clusters: str
            Column name in adata_sc.obs containing cell type labels
        markers: list
            Selected marker genes used for mapping
        ad_map: AnnData
            Mapping results after training

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load data
        >>> adata_sc = sc.read_h5ad("sc_data.h5ad")
        >>> adata_sp = sc.read_visium("spatial_data")
        >>> # Initialize Tangram
        >>> tangram = ov.space.Tangram(
        ...     adata_sc=adata_sc,
        ...     adata_sp=adata_sp,
        ...     clusters='cell_type'
        ... )
        >>> # Train model
        >>> tangram.train(mode='clusters', num_epochs=500)
        >>> # Project cell types
        >>> adata_spatial = tangram.cell2location()
    """
    def check_tangram(self):
        r"""Check if Tangram package is installed.
        
        This method verifies that the Tangram package is available and prints
        its version number. If not installed, it raises an informative error.

        Parameters
        ----------
        None

        Raises
        ------
        ImportError
            Raised when `tangram-sc` is not installed.

        Notes:
            - Sets global tg_install flag when successful
            - Required before any Tangram operations
            - Suggests pip installation command if missing
        """
        global tg_install
        try:
            import tangram as tg
            tg_install=True
            print('tangram have been install version:',tg.__version__)
        except ImportError as e:
            raise ImportError(
                'Please install the tangram: `pip install -U tangram-sc`.'
            ) from e

    def __init__(self,
            adata_sc: AnnData,
            adata_sp: AnnData,
            clusters: str = '',
            marker_size: int = 100,
            gene_to_lowercase: bool = False
        ) -> None:
        r"""Initialize Tangram spatial deconvolution object.
        
        This method sets up the Tangram analysis by:
        1. Checking package installation
        2. Processing input data
        3. Identifying marker genes
        4. Preparing data for mapping

        Parameters
        ----------
        adata_sc : AnnData
            Single-cell reference AnnData used as source profiles.
        adata_sp : AnnData
            Spatial AnnData used as target tissue map.
        clusters : str, default=''
            Cell-type annotation column in ``adata_sc.obs``.
        marker_size : int, default=100
            Number of top-ranked marker genes selected per cell type.
        gene_to_lowercase : bool, default=False
            Whether to lowercase gene names before intersection.

        Notes:
            - Automatically filters genes present in at least one cell
            - Identifies marker genes using scanpy's rank_genes_groups
            - Prepares data structures for Tangram mapping
            - Adds reference annotation to both AnnData objects
        """
        self.check_tangram()
        global tg_install
        if tg_install==True:
            global_imports("tangram","tg")
        ad_map_dict={}
        import tangram as tg

        adata_sc.uns['log1p']={}
        adata_sc.uns['log1p']['base']=None
        sc.pp.filter_genes(adata_sc, min_cells=1)
        sc.tl.rank_genes_groups(adata_sc, groupby=clusters,
                                key_added=f'{clusters}_rank_genes_groups',use_raw=False)
        markers_df = pd.DataFrame(adata_sc.uns[f"{clusters}_rank_genes_groups"]["names"]).iloc[0:marker_size, :]

        markers = list(np.unique(markers_df.melt().value.values))
        print('...Calculate The Number of Markers:',len(markers))

        self.adata_sc=adata_sc
        self.adata_sp=adata_sp
        self.clusters=clusters
        self.markers=markers

        tg.pp_adatas(self.adata_sc, self.adata_sp,
                      genes=self.markers,gene_to_lowercase=gene_to_lowercase)

        print('...Model prepared successfully')
        add_reference(self.adata_sc,'tangram','cell type classification with Tangram')
        add_reference(self.adata_sp,'tangram','cell type classification with Tangram')

    def train(self,
            mode: str = "clusters",
            num_epochs: int = 500,
            device: str = "cuda:0",
            **kwargs: Any
        ) -> None:
        r"""Train the Tangram spatial mapping model.
        
        This method trains a model to map cells or clusters from scRNA-seq data
        to spatial locations. It optimizes the mapping to preserve both gene
        expression patterns and spatial structure.

        Parameters
        ----------
        mode : {'clusters', 'cells'}, default='clusters'
            Tangram mapping mode.
        num_epochs : int, default=500
            Number of optimization epochs.
        device : str, default='cuda:0'
            Compute device used by Tangram backend.
        **kwargs : Any
            Additional arguments forwarded to ``tangram.map_cells_to_space``.

        Notes:
            - Automatically stores mapping in self.ad_map
            - Projects cell type annotations to spatial data
            - Adds reference annotation to both AnnData objects
            - Progress is shown during training
            - GPU acceleration recommended for large datasets
        """
        import tangram as tg
        ad_map = tg.map_cells_to_space(self.adata_sc, self.adata_sp,
            mode=mode,
            cluster_label=self.clusters,
            num_epochs=num_epochs,
            device=device,
            **kwargs
        )
        print(ad_map)

        tg.project_cell_annotations(ad_map, self.adata_sp, annotation=self.clusters)
        self.ad_map=ad_map
        
        print('...Model train successfully')
        add_reference(self.adata_sp,'tangram','cell type classification with Tangram')
        add_reference(self.adata_sc,'tangram','cell type classification with Tangram')

    def cell2location(self,annotation_list=None):
        r"""Project cell type annotations to spatial coordinates.
        
        This method creates a visualization-ready AnnData object containing the
        predicted cell type proportions for each spatial location.

        Parameters
        ----------
        annotation_list : list, optional
            Subset of cell types to project. Uses all discovered cell types when
            ``None``.

        Returns
        -------
        AnnData
            Spatial AnnData copy with projected cell-type proportions.

        Notes:
            - Automatically normalizes cell type proportions
            - Clips extreme values for better visualization
            - Adds reference annotation to both AnnData objects
            - Results can be directly used for spatial plotting
        """
        adata_plot=self.adata_sp.copy()
        if annotation_list is None:
            annotation_list=list(set(self.adata_sc.obs[self.clusters]))

        df = adata_plot.obsm["tangram_ct_pred"][annotation_list]
        construct_obs_plot(df, adata_plot, perc=0)
        add_reference(self.adata_sp,'tangram','cell type classification with Tangram')
        add_reference(self.adata_sc,'tangram','cell type classification with Tangram')
        return adata_plot
    
    def impute(self,
               ad_map: AnnData = None,
               ad_sc: AnnData = None,
                **kwargs: Any) -> AnnData:
        r"""Impute gene expression in spatial data using trained model.
        
        This method uses the trained mapping to predict the expression of all genes
        in the spatial locations, including genes not used in the mapping.

        Parameters
        ----------
        ad_map : AnnData, optional
            Tangram mapping AnnData from ``train``. Defaults to ``self.ad_map``.
        ad_sc : AnnData, optional
            Single-cell reference AnnData. Defaults to ``self.adata_sc``.
        **kwargs : Any
            Additional arguments forwarded to ``tangram.project_genes``.

        Returns
        -------
        AnnData
            Spatial-gene imputation AnnData returned by Tangram.

        Notes:
            - Uses mapping weights to transfer expression
            - Can impute genes not used in original mapping
            - Useful for analyzing spatial patterns of any gene
            - Computationally intensive for large gene sets
        """
        import tangram as tg

        if ad_map is None:
            ad_map=self.ad_map
        if ad_sc is None:
            ad_sc=self.adata_sc
        ad_ge = tg.project_genes(adata_map=ad_map, 
                                 adata_sc=ad_sc,**kwargs)
        return ad_ge


def construct_obs_plot(df_plot: pd.DataFrame,
                        adata: AnnData,
                        perc: int = 0,
                        suffix = None
                    ) -> None:
    r"""Construct observation metadata from plotting DataFrame.
    
    This helper function processes cell type proportion data for visualization
    by normalizing and optionally clipping extreme values.

    Parameters
    ----------        df_plot: pd.DataFrame
            DataFrame containing cell type proportions or other values
            to be added to observation metadata.
        adata: AnnData
            AnnData object to update with processed values.
        perc: int, optional (default=0)
            Percentile for clipping values. Values outside
            (perc, 100-perc) are clipped. Use 0 for no clipping.
        suffix: str, optional (default=None)
            Optional suffix to add to column names in the output.
            Useful when storing multiple versions of the same metric.

    Notes:
        - Clips values to remove outliers if perc > 0
        - Normalizes values to [0,1] range
        - Adds processed values to adata.obs
        - Preserves existing observation metadata
    """
    # clip
    df_plot = df_plot.clip(df_plot.quantile(perc), df_plot.quantile(1 - perc), axis=1)

    # Normalize varying columns. A constant prediction is already finite and
    # meaningful; dividing by a zero range used to turn the entire column NaN.
    minima = df_plot.min()
    ranges = df_plot.max() - minima
    varying = ranges > 0
    normalized = df_plot.copy()
    if varying.any():
        normalized.loc[:, varying] = (
            df_plot.loc[:, varying] - minima.loc[varying]
        ) / ranges.loc[varying]
    df_plot = normalized

    if suffix:
        df_plot = df_plot.add_suffix(" ({})".format(suffix))
    adata.obs = pd.concat([adata.obs, df_plot], axis=1)

def global_imports(modulename,shortname = None, asfunction = False):
    r"""Import a module and add it to the global namespace.
    
    This helper function dynamically imports a module and makes it available
    in the global namespace, optionally with a custom name.

    Parameters
    ----------        modulename: str
            Name of the module to import (e.g., 'numpy').
        shortname: str, optional (default=None)
            Alternative name to use in the global namespace.
            If None, uses the module name.
        asfunction: bool, optional (default=False)
            Whether to import as a function rather than a module.
            Rarely needed for standard module imports.

    Notes:
        - Modifies the global namespace
        - Use with caution to avoid naming conflicts
        - Primarily used for dynamic package loading
        - Consider using standard imports when possible
    """
    if shortname is None:
        shortname = modulename
    if asfunction is False:
        globals()[shortname] = __import__(modulename)
    else:
        globals()[shortname] = __import__(modulename)
    # End-of-file (EOF)
