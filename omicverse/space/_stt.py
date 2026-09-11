r"""Module providing encapsulation of STT for spatial transition tensor analysis."""
from typing import Any
import warnings
import scanpy as sc
import numpy as np
import pandas as pd
from .._settings import add_reference
from .._registry import register_function


def _get_stt_modules():
    from ..external.STT import pl, tl

    return tl, pl

@register_function(
    aliases=["STT空间转换张量", "STT", "spatial_transition_tensor", "空间转换分析", "空间动力学"],
    category="space",
    description="Spatial Transition Tensor analysis for modeling spatial dynamics and cell state transitions",
    prerequisites={
        'functions': []  # Requires RNA velocity preprocessing (spliced/unspliced)
    },
    requires={
        'layers': ['spliced', 'unspliced'],  # RNA velocity data required
        'obsm': []  # Spatial coordinates (user-specified)
    },
    produces={
        'layers': ['velocity'],
        'obs': ['pseudotime']
    },
    auto_fix='escalate',
    examples=[
        "# Basic STT analysis",
        "stt = ov.space.STT(adata, spatial_loc='spatial',",
        "                   region='tissue_region')",
        "# Stage estimation",
        "stt.stage_estimate()",
        "# Train STT model",
        "stt.train(n_states=10)",
        "# Custom parameters",
        "stt = ov.space.STT(adata, spatial_loc='xy_loc',",
        "                   region='Region')"
    ],
    related=["space.svg", "space.clusters", "external.scvelo"]
)
class STT(object):
    r"""Spatial Transition Tensor (STT) analysis class.
    
    STT models spatial dynamics and transitions by learning spatial-temporal patterns
    in spatial transcriptomics data using transition tensors. This class provides methods
    for analyzing cell state transitions, developmental trajectories, and spatial dynamics
    in tissue organization.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData containing spliced/unspliced layers and coordinates.
    spatial_loc : str, optional
        Coordinate key in ``adata.obsm``. When omitted, the legacy ``'xy_loc'``
        key is retained if present; otherwise the conventional ``'spatial'``
        key is used.
    region : str, default='Region'
        Region annotation column in ``adata.obs``.

    Attributes:
        adata: AnnData
            Input annotated data matrix.
        adata_aggr: AnnData or None
            Aggregated data after training.
        spatial_loc: str
            Key for spatial coordinates.
        region: str
            Key for region annotations.

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load data with spliced/unspliced counts
        >>> adata = sc.read_h5ad('spatial_velocity.h5ad')
        >>> # Initialize STT object
        >>> stt = ov.space.STT(
        ...     adata,
        ...     spatial_loc='spatial',
        ...     region='tissue_region'
        ... )
        >>> # Estimate cell stages
        >>> stt.stage_estimate()
        >>> # Train the model
        >>> stt.train(n_states=10)
    """
    def __init__(self,adata,spatial_loc=None,region='Region'):
        r"""Initialize STT spatial transition analysis object.
        
        Parameters
        ----------
        adata : AnnData
            Input AnnData with spatial and velocity-related layers.
        spatial_loc : str, optional
            Coordinate key in ``adata.obsm``. Omitting it preserves the legacy
            ``'xy_loc'`` default when available, with a migration warning, and
            otherwise uses ``'spatial'``.
        region : str, default='Region'
            Region key in ``adata.obs``.
        """
        if spatial_loc is None:
            if 'xy_loc' in adata.obsm:
                spatial_loc = 'xy_loc'
                warnings.warn(
                    "STT is using the legacy default adata.obsm['xy_loc']. Pass "
                    "spatial_loc='spatial' explicitly to use standard AnnData "
                    "spatial coordinates; the automatic legacy preference will be "
                    "removed in a future release.",
                    FutureWarning,
                    stacklevel=2,
                )
            elif 'spatial' in adata.obsm:
                spatial_loc = 'spatial'
            else:
                raise KeyError(
                    "Neither adata.obsm['xy_loc'] nor adata.obsm['spatial'] exists; "
                    "pass `spatial_loc` explicitly."
                )
        if spatial_loc not in adata.obsm:
            raise KeyError(f"spatial_loc {spatial_loc!r} was not found in adata.obsm.")
        if region not in adata.obs:
            raise KeyError(f"region {region!r} was not found in adata.obs.")
        self.adata=adata
        self.adata_aggr=None
        self.spatial_loc=spatial_loc
        self.region=region
        if 'attractor' not in self.adata.obs.keys():
            self.adata.obs['attractor'] = self.adata.obs[region]

    def stage_estimate(self):
        r"""Estimate cell stages using joint clustering of spliced and unspliced data.
        
        This method performs joint analysis of spliced and unspliced RNA counts to identify
        developmental stages through clustering. It uses a combination of PCA, nearest
        neighbor graph construction, and Leiden clustering to identify distinct cell stages.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Writes stage labels to ``adata.obs['joint_leiden']``.

        Notes:
            - Requires spliced/unspliced counts in adata.layers
            - Uses resolution=0.15 for Leiden clustering
            - Number of stages is determined automatically
            - Results are stored in adata.obs['joint_leiden']
        """
        u = self.adata.layers['unspliced']
        s = self.adata.layers['spliced']
        if 'toarray' in dir(u):
            u = u.toarray()
            s = s.toarray()
        x_all = np.concatenate((u,s),axis = 1)
        adata_aggr = sc.AnnData(X=x_all)
        sc.tl.pca(adata_aggr, svd_solver='arpack')
        sc.pp.neighbors(adata_aggr)
        sc.tl.leiden(adata_aggr,resolution = 0.15)
        self.adata.obs['joint_leiden'] = adata_aggr.obs['leiden'].values
        print(f"...estimate stage: {len(self.adata.obs['joint_leiden'].unique())}")
        add_reference(self.adata,'STT','spatial transition tensor with STT')

    def train(self,
            n_states: int = 9,
            n_iter: int = 15,
            weight_connectivities: float = 0.5,
            n_neighbors: int = 50,
            thresh_ms_gene: float = 0.2,
            spa_weight: float = 0.3,
            **kwargs: Any
        ) -> None:
        r"""Train the STT spatial transition model.
        
        This method trains the Spatial Transition Tensor model to learn spatial-temporal
        patterns and cell state transitions. It combines connectivity constraints with
        spatial information to model developmental trajectories.

        Parameters
        ----------
        n_states : int, default=9
            Number of latent transition states.
        n_iter : int, default=15
            Iterations for dynamical optimization.
        weight_connectivities : float, default=0.5
            Connectivity-term weight.
        n_neighbors : int, default=50
            Neighbors used in graph construction.
        thresh_ms_gene : float, default=0.2
            Threshold used in marker/feature filtering.
        spa_weight : float, default=0.3
            Spatial-term weight.
        **kwargs : Any
            Additional options passed to ``tl.dynamical_iteration``.

        Returns
        -------
        None
            Stores aggregated STT result in ``self.adata_aggr``.

        Notes:
            - Higher spa_weight emphasizes spatial relationships
            - Higher weight_connectivities emphasizes transcriptional similarity
            - Results are stored in self.adata_aggr
            - Spatial coordinates are preserved in obsm
        """
        tl, _ = _get_stt_modules()
        self.adata_aggr = tl.dynamical_iteration(self.adata,n_states = n_states,
                                    n_iter = n_iter, weight_connectivities = weight_connectivities,
                                    n_neighbors = n_neighbors,thresh_ms_gene = thresh_ms_gene,
                                    spa_weight =spa_weight,**kwargs)
        self.adata.obsm[f'X_{self.spatial_loc}'] = self.adata.obsm[self.spatial_loc]
        self.adata_aggr.obsm[f'X_{self.spatial_loc}']=self.adata.obsm[self.spatial_loc]
        self.adata_aggr.obsm[f'X_{self.spatial_loc}_aggr']=self.adata.obsm[self.spatial_loc]
        self.adata.obsm[f'X_{self.spatial_loc}_aggr']=self.adata.obsm[self.spatial_loc]
        add_reference(self.adata,'STT','spatial transition tensor with STT')

    def load(self,adata,adata_aggr):
        r"""Load pre-trained STT model data.
        
        This method allows loading previously trained STT model results, enabling
        analysis continuation or result sharing.

        Parameters
        ----------
        adata : AnnData
            Original spatial AnnData.
        adata_aggr : AnnData
            Aggregated STT result AnnData.

        Returns
        -------
        None
            Updates internal references to loaded objects.

        Notes:
            - Useful for sharing analysis results
            - Preserves all STT-specific annotations
            - Maintains spatial coordinate information
        """
        self.adata=adata
        self.adata_aggr=adata_aggr
        add_reference(self.adata,'STT','spatial transition tensor with STT')

    def compute_pathway(self,pathway_dict,n_components=10):
        r"""Compute spatial pathways for cell transitions.
        
        This method identifies and computes transition pathways between cell states
        in spatial context, revealing developmental trajectories and cell fate decisions.

        Parameters
        ----------
        pathway_dict : dict
            Pathway specification for STT pathway computation.
        n_components : int, default=10
            Number of decomposition components for pathway computation.

        Returns
        -------
        dict
            Pathway computation result from STT backend.

        Notes:
            - Pathways are computed using transition tensors
            - Results can be visualized using plot_pathway()
            - Considers both spatial and transcriptional information
        """
        tl, _ = _get_stt_modules()
        return tl.compute_pathway(self.adata,self.adata_aggr,pathway_dict,n_components=n_components)

    def plot_pathway(self,label_fontsize=20,**kwargs):
        r"""Plot spatial transition pathways.
        
        This method visualizes computed transition pathways in spatial context,
        showing how cells transition between states while maintaining spatial organization.

        Parameters
        ----------
        label_fontsize : int, default=20
            Axis label font size.
        **kwargs : Any
            Extra options passed to STT plotting backend.

        Returns
        -------
        matplotlib.figure.Figure
            Pathway visualization figure.

        Notes:
            - Multiple pathways can be visualized simultaneously
            - Colors indicate transition states
            - Arrows show direction of transitions
            - Spatial coordinates are preserved in visualization
        """
        _, pl = _get_stt_modules()
        fig = pl.plot_pathway(self.adata,**kwargs)
        for ax in fig.axes:
            ax.set_xlabel('Embedding 1', fontsize=label_fontsize)
            ax.set_ylabel('Embedding 2', fontsize=label_fontsize)
        return fig

    def plot_tensor_pathway(self,pathway_name,**kwargs):
        r"""Plot tensor-based spatial pathways.
        
        This method creates a specialized visualization of transition pathways using
        tensor decomposition results, highlighting the continuous nature of cell state
        transitions in space.

        Parameters
        ----------
        pathway_name : str
            Name of pathway to visualize.
        **kwargs : Any
            Extra tensor-pathway plotting arguments.

        Returns
        -------
        matplotlib.axes.Axes
            Tensor-pathway axes.

        Notes:
            - Tensor visualization shows transition probabilities
            - Thickness of edges indicates transition likelihood
            - Colors represent different cell states
            - Spatial arrangement reflects tissue organization
        """
        _, pl = _get_stt_modules()
        ax=pl.plot_tensor_pathway(self.adata,self.adata_aggr,
                                  pathway_name = pathway_name,**kwargs)
        return ax

    def plot_tensor(self,list_attractor,**kwargs):
        r"""Plot spatial transition tensors.
        
        This method visualizes the learned transition tensors, showing how cells
        transition between states while considering spatial constraints.

        Parameters
        ----------
        list_attractor : list
            Attractor/state list to visualize.
        **kwargs : Any
            Extra arguments for tensor plotting.

        Returns
        -------
        dict
            Plotting objects/metadata returned by STT backend.

        Notes:
            - Tensors show transition probabilities between states
            - Higher values indicate more likely transitions
            - Spatial relationships are encoded in tensor structure
            - Can be used to identify major transition paths
        """
        _, pl = _get_stt_modules()
        return pl.plot_tensor(self.adata, self.adata_aggr, 
                          list_attractor = list_attractor,basis = self.spatial_loc,**kwargs)

    def construct_landscape(self,coord_key=None,**kwargs):
        r"""Construct spatial landscape for transition analysis.
        
        This method builds a continuous landscape representation of cell state transitions
        in spatial context, useful for understanding developmental potential and barriers.

        Parameters
        ----------
        coord_key : str, optional
            Coordinate key used for landscape construction. Defaults to the
            trained key corresponding to ``spatial_loc``.
        **kwargs : Any
            Additional backend options for landscape construction.

        Returns
        -------
        None
            Writes landscape coordinates to ``adata.obsm['trans_coord']``.

        Notes:
            - Landscape represents continuous transition space
            - Valleys indicate stable states
            - Ridges represent transition barriers
            - Coordinates preserve both spatial and state information
        """
        tl, _ = _get_stt_modules()
        if coord_key is None:
            coord_key = f'X_{self.spatial_loc}'
        tl.construct_landscape(self.adata, coord_key = coord_key,**kwargs)
        self.adata.obsm['trans_coord'] = self.adata.uns['land_out']['trans_coord']

    def infer_lineage(self,**kwargs):
        r"""Infer cell lineage relationships from spatial transitions.
        
        This method reconstructs developmental lineages by analyzing transition
        patterns and spatial relationships between cell states.

        Parameters
        ----------
        **kwargs : Any
            Additional lineage-inference options.

        Returns
        -------
        dict
            Inferred lineage result.

        Notes:
            - Combines spatial and transcriptional information
            - Identifies branching points in development
            - Considers transition probabilities between states
            - Results can be visualized with plot_landscape()
        """
        _, pl = _get_stt_modules()
        return pl.infer_lineage(self.adata,**kwargs)

    def plot_landscape(self,**kwargs):
        r"""Plot spatial landscape visualization.
        
        This method creates a visual representation of the constructed transition
        landscape, showing the continuous spectrum of cell states in spatial context.

        Parameters
        ----------
        **kwargs : Any
            Additional landscape plotting arguments.

        Returns
        -------
        matplotlib.figure.Figure
            Landscape plot.

        Notes:
            - Colors indicate cell states or features
            - Contours show transition barriers
            - Arrows indicate preferred transition directions
            - Spatial relationships are preserved in layout
        """
        _, pl = _get_stt_modules()
        return pl.plot_landscape(self.adata,**kwargs)

    def plot_sankey(self,vector1, vector2):
        r"""Plot Sankey diagram for cell transitions.
        
        This method creates a Sankey diagram showing the flow of cells between
        different states or conditions, useful for understanding transition dynamics.

        Parameters
        ----------
        vector1 : array-like
            Source-state labels.
        vector2 : array-like
            Target-state labels.

        Returns
        -------
        matplotlib.figure.Figure
            Sankey figure.

        Notes:
            - Width of flows indicates transition frequency
            - Colors can represent different states
            - Useful for visualizing state changes
            - Shows conservation of cell numbers
        """
        _, pl = _get_stt_modules()
        return pl.plot_sankey(vector1, vector2)

    def plot_top_genes(self,**kwargs):
        r"""Plot top genes driving spatial transitions.
        
        This method identifies and visualizes genes that are most important in
        determining cell state transitions and spatial patterns.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments for top-gene plotting.

        Returns
        -------
        matplotlib.figure.Figure
            Top-gene visualization.

        Notes:
            - Highlights genes driving state transitions
            - Shows expression patterns across states
            - Useful for identifying key regulators
            - Can reveal transition mechanisms
        """
        _, pl = _get_stt_modules()
        return pl.plot_top_genes(self.adata,**kwargs)
    # End-of-file (EOF)
