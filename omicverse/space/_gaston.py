import os
import torch
import numpy as np
from scipy.sparse import issparse, csr_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from .._settings import add_reference
from .._registry import register_function


def _get_gaston_modules():
    from ..external.gaston import cluster_plotting, dp_related, neural_net, process_NN_output

    return neural_net, process_NN_output, dp_related, cluster_plotting

@register_function(
    aliases=["GASTON空间深度分析", "GASTON", "spatial_depth_analysis", "空间深度建模", "组织空间组织"],
    category="space",
    description="GASTON spatial depth estimation and clustering using neural networks",
    prerequisites={
        'functions': [],
        'optional_functions': []
    },
    requires={
        'obsm': ['spatial']
    },
    produces={
        'uns': ['gaston'],
        'obs': ['gaston_isodepth', 'gaston_labels']
    },
    auto_fix='none',
    examples=[
        "# Basic GASTON analysis",
        "gaston = ov.space.GASTON(adata)",
        "counts, coords, genes = gaston.get_gaston_input()",
        "features = gaston.get_top_pearson_residuals(num_dims=10)",
        "gaston.load_rescale(features)",
        "gaston.train(num_epochs=5000)",
        "# Get results",
        "model, features, coords = gaston.get_best_model()",
        "isodepth, labels = gaston.cal_iso_depth(num_domains=5)",
        "# With RGB features",
        "counts, coords, genes, rgb = gaston.get_gaston_input(get_rgb=True)",
        "# Plotting results",
        "gaston.plot_gaston_scatter()"
    ],
    related=["space.clusters", "space.svg", "space.pySTAGATE"]
)
class GASTON(object):
    r"""GASTON spatial depth estimation and clustering.
    
    GASTON (Geometry And Spatial Transcriptomics-based OrganizatioN) is a method
    for analyzing spatial transcriptomics data by learning continuous spatial depth
    functions that capture tissue organization. It uses neural networks to model
    spatial patterns and identify distinct domains.

    The method combines gene expression data with spatial information to:
    1. Learn continuous spatial depth functions
    2. Identify tissue domains and boundaries
    3. Model gene expression patterns along spatial gradients
    4. Characterize tissue organization and architecture

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData containing expression matrix and coordinates in
        ``adata.obsm['spatial']``.

    Attributes:
        adata: AnnData
            Input annotated data matrix containing:
            - Spatial coordinates in adata.obsm['spatial']
            - Gene expression data in adata.X
            - Optional histology image in adata.uns['spatial']
        model: GASTON model
            Trained neural network model after calling train()
        gaston_isodepth: array
            Computed isodepth values after calling cal_iso_depth()
        gaston_labels: array
            Domain labels after calling cal_iso_depth()

    Examples:
        >>> import omicverse as ov
        >>> # Load spatial data
        >>> adata = ov.io.read_visium(...)
        >>> # Initialize GASTON
        >>> gaston = ov.space.GASTON(adata)
        >>> # Prepare input data
        >>> counts, coords, genes = gaston.get_gaston_input()
        >>> # Get features
        >>> features = gaston.get_top_pearson_residuals(num_dims=10)
        >>> # Train model
        >>> gaston.load_rescale(features)
        >>> gaston.train(num_epochs=5000)
        >>> # Get best model
        >>> model, features, coords = gaston.get_best_model()
        >>> # Calculate domains
        >>> isodepth, labels = gaston.cal_iso_depth(num_domains=5)
    """

    def __init__(self,adata) -> None:
        r"""Initialize GASTON spatial clustering object.
        
        Parameters
        ----------
        adata : AnnData
            Input spatial AnnData.
        """
        self.adata=adata

    def get_gaston_input(self,get_rgb=False, spot_umi_threshold=50):
        r"""Prepare input data for GASTON analysis.
        
        This method processes the input AnnData object to extract necessary data
        for GASTON analysis, including gene expression counts, spatial coordinates,
        and optionally RGB features from histology images.

        Parameters
        ----------
        get_rgb : bool, default=False
            Whether to extract per-spot RGB features from tissue image.
        spot_umi_threshold : int, default=50
            Minimum total counts per spot for filtering.

        Returns
        -------
        tuple
            ``(counts_mat, coords_mat, gene_labels)`` or additionally
            ``RGB_mean`` when ``get_rgb=True``.

        Notes:
            - Filters spots based on UMI count threshold
            - RGB features are extracted from high-resolution images if available
            - RGB values are normalized to [0,1] range
            - Coordinates are converted to float type
        """
        import squidpy as sq
        adata=self.adata
        adata.obsm['spatial']=adata.obsm['spatial'].astype(float)
        from ..pp import filter_cells
        filter_cells(adata, min_counts=spot_umi_threshold)

        gene_labels=adata.var.index.to_numpy()

        counts_mat=adata.X
        coords_mat=np.array(adata.obsm['spatial'])

        if not get_rgb:
            return counts_mat, coords_mat, gene_labels

        library_id = list(adata.uns['spatial'].keys())[0] # adata2.uns['spatial'] should have only one key
        scale=adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
        img = sq.im.ImageContainer(adata.uns['spatial'][library_id]['images']['hires'],
                                scale=scale,
                                layer="img1")
        print('calculating RGB')
        sq.im.calculate_image_features(adata, img, features="summary", key_added="features")
        columns = ['summary_ch-0_mean', 'summary_ch-1_mean', 'summary_ch-2_mean']
        RGB_mean = adata.obsm["features"][columns]
        RGB_mean = RGB_mean / 255
        self.RGB_mean=RGB_mean
        
        adata=self.adata
        return counts_mat, coords_mat, gene_labels, RGB_mean.to_numpy()
    
    def get_top_pearson_residuals(self,num_dims=5,clip=0.01,n_top_genes=5000,
                                  use_RGB=False):
        r"""Get top Pearson residual features for GASTON analysis.
        
        This method selects highly variable genes using Pearson residuals,
        performs dimensionality reduction, and optionally combines with RGB features.

        Parameters
        ----------
        num_dims : int, default=5
            Number of PCA components kept as GASTON features.
        clip : float, default=0.01
            Clipping parameter used in Pearson residual normalization.
        n_top_genes : int, default=5000
            Number of highly variable genes considered.
        use_RGB : bool, default=False
            Whether to concatenate RGB features (requires ``get_rgb=True`` before).

        Returns
        -------
        numpy.ndarray
            Feature matrix for GASTON training.

        Notes:
            - Uses Pearson residuals for robust feature selection
            - Applies sqrt-normalization before PCA
            - RGB features are concatenated after PCA if use_RGB=True
            - Features are used as input for neural network training
        """
        from ..pp import normalize_total
        from ..pp.experimental import (
            highly_variable_genes as pr_highly_variable_genes,
            normalize_pearson_residuals,
        )
        from .._sc_compat import pca_X

        adata=self.adata
        pr_highly_variable_genes(
            adata, flavor="pearson_residuals", n_top_genes=n_top_genes
        )

        adata = adata[:, adata.var["highly_variable"]]
        adata.layers["raw"] = adata.X.copy()
        adata.layers["sqrt_norm"] = np.sqrt(
            normalize_total(adata, inplace=False)["X"]
        )

        theta=np.inf
        normalize_pearson_residuals(adata, clip=clip, theta=theta)
        pca_X(adata, n_comps=num_dims)
        A=adata.obsm['X_pca']
        if use_RGB:
            A=np.hstack((A,self.RGB_mean)) # attach to RGB mean
        return A
    
    def load_rescale(self,A):
        r"""Load and rescale input data for neural network training.
        
        This method prepares the feature matrix and spatial coordinates for
        neural network training by rescaling them to appropriate ranges.

        Parameters
        ----------
        A : numpy.ndarray
            Feature matrix from ``get_top_pearson_residuals``.

        Notes:
            - Converts data to PyTorch tensors
            - Rescales features to standardized range
            - Stores results in self.S_torch and self.A_torch
            - Required before calling train()
        """
        S=self.adata.obsm['spatial']
        from ..external.gaston.neural_net import load_rescale_input_data
        S_torch, A_torch = load_rescale_input_data(S,A)
        self.S_torch=S_torch
        self.A_torch=A_torch
    
    def train(self,isodepth_arch=[20,20],expression_fn_arch=[20,20],
              num_epochs=10000,checkpoint=500,out_dir='result/test_outputs',
              optimizer="adam",num_restarts=30):
        r"""Train GASTON neural network models.
        
        This method trains neural networks to learn spatial depth functions and
        gene expression patterns. It performs multiple training runs with different
        random initializations to ensure robust results.

        Parameters
        ----------
        isodepth_arch : list, default=[20, 20]
            Hidden-layer architecture for depth network.
        expression_fn_arch : list, default=[20, 20]
            Hidden-layer architecture for expression network.
        num_epochs : int, default=10000
            Epochs per restart.
        checkpoint : int, default=500
            Checkpoint interval.
        out_dir : str, default='result/test_outputs'
            Output directory for model checkpoints.
        optimizer : str, default='adam'
            Optimizer name.
        num_restarts : int, default=30
            Number of random restarts.

        Notes:
            - Multiple restarts help avoid local optima
            - Models are saved periodically during training
            - Each restart uses a different random initialization
            - Best model can be selected using get_best_model()
            - Training progress is shown with a progress bar
        """
        os.makedirs(out_dir, exist_ok=True)
        neural_net, _, _, _ = _get_gaston_modules()

        seed_list=range(num_restarts)
        for seed in tqdm(seed_list):
            out_dir_seed=f"{out_dir}/rep{seed}"
            os.makedirs(out_dir_seed, exist_ok=True)
            mod, loss_list = neural_net.train(self.S_torch, self.A_torch,
                                S_hidden_list=isodepth_arch, A_hidden_list=expression_fn_arch, 
                                epochs=num_epochs, checkpoint=checkpoint, 
                                save_dir=out_dir_seed, optim=optimizer, seed=seed, save_final=True)
            
        add_reference(self.adata,'GASTON','spatial depth estimation with GASTON')

    def get_best_model(self,out_dir='result/test_outputs',
                       max_domain_num=8,start_from=2):
        r"""Select best GASTON model and determine optimal domain number.
        
        This method analyzes trained models to select the best performing one
        and helps determine the optimal number of spatial domains.

        Parameters
        ----------
        out_dir : str, default='result/test_outputs'
            Directory containing saved restart models.
        max_domain_num : int, default=8
            Maximum domain count tested during model selection.
        start_from : int, default=2
            Minimum domain count tested.

        Returns
        -------
        tuple
            ``(best_model, features, coordinates)``.

        Notes:
            - Uses likelihood curves to determine optimal domain number
            - Plots model selection diagnostics
            - Stores selected model in self.model
            - Required before calling cal_iso_depth()
        """
        _, process_NN_output, _, _ = _get_gaston_modules()
        gaston_model, A, S= process_NN_output.process_files(out_dir)
        self.model=gaston_model
        self.A=A
        self.S=S
        from ..external.gaston import model_selection
        model_selection.plot_ll_curve(gaston_model, A, S, max_domain_num=max_domain_num, start_from=start_from)
        return gaston_model, A, S
    
    def cal_iso_depth(self,num_domains=10):
        r"""Calculate isodepth values and domain labels.
        
        This method uses the trained model to compute continuous spatial depth
        values and assign spots to discrete spatial domains.

        Parameters
        ----------            num_domains: int, optional (default=10)
                Number of spatial domains to identify.
            
        Returns
        -------            tuple
                (isodepth_values, domain_labels):
                - isodepth_values: Continuous spatial depth per spot
                - domain_labels: Discrete domain assignments

        Notes:
            - Isodepth values represent continuous spatial organization
            - Domain labels are discrete assignments (0 to num_domains-1)
            - Results are stored in self.gaston_isodepth and self.gaston_labels
            - Values are adjusted so higher numbers indicate deeper layers
        """
        _, _, dp_related, _ = _get_gaston_modules()
        gaston_isodepth, gaston_labels=dp_related.get_isodepth_labels(self.model,
                                                                      self.A,
                                                                      self.S,
                                                                      num_domains)

        # Adjust so higher values indicate deeper layers
        gaston_isodepth = np.max(gaston_isodepth) -1 * gaston_isodepth
        gaston_labels = (num_domains-1) - gaston_labels
        self.gaston_isodepth = gaston_isodepth
        self.gaston_labels = gaston_labels
        return gaston_isodepth, gaston_labels
    
    def plot_isodepth(self,show_streamlines=True,
                      rotate_angle=-90,arrowsize=2,
                      figsize=(7,6),**kwargs):
        r"""Plot isodepth contours and streamlines.
        
        This method creates visualizations of the learned spatial organization
        including isodepth contours and optional flow streamlines.

        Parameters
        ----------            show_streamlines: bool, optional (default=True)
                Whether to show directional flow streamlines.
            rotate_angle: float, optional (default=-90)
                Rotation angle in degrees for plot orientation.
            arrowsize: float, optional (default=2)
                Size of streamline arrows.
            figsize: tuple, optional (default=(7,6))
                Figure size in inches.
            **kwargs:
                Additional arguments passed to plotting functions.

        Returns
        -------            matplotlib.figure.Figure
                Figure containing the visualization.

        Notes:
            - Contours show lines of constant spatial depth
            - Streamlines indicate gradient directions
            - Colors represent depth values
            - Useful for understanding tissue organization
        """
        from ..external.gaston.cluster_plotting import plot_isodepth
        rotate=np.radians(rotate_angle)
        plot_isodepth(self.gaston_isodepth,self.S, 
                      self.model, figsize=figsize, streamlines=show_streamlines, 
                      rotate=rotate,arrowsize=arrowsize,**kwargs) # since we did isodepth -> -1*isodepth above, we also need to do gradient -> -1*gradient

    def plot_clusters(self,domain_colors,figsize=(6,6),
                      s=20,lgd=False,show_boundary=True,
                      rotate_angle=-90,boundary_lw=5,**kwargs):
        r"""Plot spatial domains with cluster colors.
        
        This method visualizes the identified spatial domains by coloring spots
        according to their domain assignments.

        Parameters
        ----------            domain_colors: dict or list
                Colors for each spatial domain.
            figsize: tuple, optional (default=(6,6))
                Figure size in inches.
            s: float, optional (default=20)
                Size of spots in scatter plot.
            lgd: bool, optional (default=False)
                Whether to show legend.
            show_boundary: bool, optional (default=True)
                Whether to show domain boundaries.
            rotate_angle: float, optional (default=-90)
                Rotation angle in degrees for plot orientation.
            boundary_lw: float, optional (default=5)
                Line width for domain boundaries.
            **kwargs:
                Additional arguments passed to plotting functions.

        Returns
        -------            matplotlib.figure.Figure
                Figure containing the domain visualization.

        Notes:
            - Each domain is shown in a different color
            - Boundaries between domains can be highlighted
            - Spots are colored by their domain assignment
            - Useful for visualizing discrete spatial structure
        """
        _, _, _, cluster_plotting = _get_gaston_modules()
        rotate = np.radians(rotate_angle)
        cluster_plotting.plot_clusters(self.gaston_labels, self.S, figsize=figsize, 
                               colors=domain_colors, s=s, lgd=lgd, 
                               show_boundary=show_boundary, 
                               gaston_isodepth=self.gaston_isodepth, 
                               boundary_lw=boundary_lw, rotate=rotate,
                               **kwargs)
        
    def plot_clusters_restrict(self,domain_colors,isodepth_min=4.5,isodepth_max=6.8,
                               rotate_angle=-90,s=20,lgd=False,figsize=(6,6), **kwargs):
        r"""Plot spatial domains with restricted isodepth range.
        
        This method visualizes spatial domains while focusing on a specific range
        of isodepth values, useful for examining particular tissue layers.

        Parameters
        ----------            domain_colors: dict or list
                Colors for each spatial domain.
            isodepth_min: float, optional (default=4.5)
                Minimum isodepth value to include.
            isodepth_max: float, optional (default=6.8)
                Maximum isodepth value to include.
            rotate_angle: float, optional (default=-90)
                Rotation angle in degrees for plot orientation.
            s: float, optional (default=20)
                Size of spots in scatter plot.
            lgd: bool, optional (default=False)
                Whether to show legend.
            figsize: tuple, optional (default=(6,6))
                Figure size in inches.
            **kwargs:
                Additional arguments passed to plotting functions.

        Returns
        -------            matplotlib.figure.Figure
                Figure containing the restricted domain visualization.

        Notes:
            - Only shows spots within specified isodepth range
            - Useful for focusing on specific tissue layers
            - Maintains domain color scheme from plot_clusters()
            - Helps visualize layer-specific patterns
        """
        _, _, _, cluster_plotting = _get_gaston_modules()
        rotate = np.radians(rotate_angle)
        cluster_plotting.plot_clusters_restrict(self.gaston_labels, self.S, self.gaston_isodepth, 
                                                isodepth_min=isodepth_min, isodepth_max=isodepth_max, figsize=figsize, 
                                                colors=domain_colors, s=s, lgd=lgd, rotate=rotate, **kwargs)
        
    def restrict_spot(self,isodepth_min=4.5,isodepth_max=6.8,
                      adjust_physical=True,scale_factor=100,
                      plotisodepth=True,show_streamlines=True,
                      rotate_angle=-90,arrowsize=1, figsize=(6,3), 
                      neg_gradient=True,
                      **kwargs):
        r"""Restrict analysis to spots within specific isodepth range.
        
        This method filters spots based on isodepth values and optionally adjusts
        for physical distances, useful for layer-specific analyses.

        Parameters
        ----------            isodepth_min: float, optional (default=4.5)
                Minimum isodepth value to include.
            isodepth_max: float, optional (default=6.8)
                Maximum isodepth value to include.
            adjust_physical: bool, optional (default=True)
                Whether to adjust for physical distances.
            scale_factor: float, optional (default=100)
                Scale factor for physical distance adjustment.
            plotisodepth: bool, optional (default=True)
                Whether to plot isodepth contours.
            show_streamlines: bool, optional (default=True)
                Whether to show gradient streamlines.
            rotate_angle: float, optional (default=-90)
                Rotation angle in degrees for plot orientation.
            arrowsize: float, optional (default=1)
                Size of streamline arrows.
            figsize: tuple, optional (default=(6,3))
                Figure size in inches.
            neg_gradient: bool, optional (default=True)
                Whether to negate gradient direction.
            **kwargs:
                Additional arguments passed to plotting functions.

        Returns
        -------            tuple
                (counts, coords, isodepth, labels, features):
                - counts: Filtered count matrix
                - coords: Filtered spatial coordinates
                - isodepth: Filtered isodepth values
                - labels: Filtered domain labels
                - features: Filtered feature matrix

        Notes:
            - Results are stored in object attributes with '_restrict' suffix
            - Physical distance adjustment helps maintain spatial relationships
            - Visualization shows restricted region context
            - Useful for focused analysis of specific tissue layers
        """
        rotate = np.radians(rotate_angle)

        from ..external.gaston.restrict_spots import restrict_spots
        counts_mat_restrict, coords_mat_restrict, gaston_isodepth_restrict, gaston_labels_restrict, S_restrict=restrict_spots(
                                                                    self.adata.X, 
                                                                    self.adata.obsm['spatial'], 
                                                                    self.S, self.gaston_isodepth, self.gaston_labels, 
                                                                    isodepth_min=isodepth_min, isodepth_max=isodepth_max, 
                                                                    adjust_physical=adjust_physical, scale_factor=scale_factor,
                                                                    plotisodepth=plotisodepth, show_streamlines=show_streamlines, 
                                                                    gaston_model=self.model, rotate=rotate, figsize=figsize, 
                                                                    arrowsize=arrowsize, 
                                                                    neg_gradient=neg_gradient,**kwargs)
        self.counts_mat_restrict=counts_mat_restrict
        self.coords_mat_restrict=coords_mat_restrict
        self.gaston_isodepth_restrict=gaston_isodepth_restrict
        self.gaston_labels_restrict=gaston_labels_restrict
        self.S_restrict=S_restrict
        # for get_restricted_adata
        self.locs=np.array( [i for i in range(len(self.gaston_isodepth)) if isodepth_min < self.gaston_isodepth[i] < isodepth_max] )

        return counts_mat_restrict, coords_mat_restrict, gaston_isodepth_restrict, gaston_labels_restrict, S_restrict
    
    def filter_genes(self,umi_thresh = 1000,exclude_prefix=['Mt-', 'Rpl', 'Rps']):
        r"""Filter genes based on expression and name patterns.
        
        This method removes genes with low expression and those matching specified
        prefixes (e.g., mitochondrial and ribosomal genes).

        Parameters
        ----------            umi_thresh: int, optional (default=1000)
                Minimum total UMI count threshold for genes.
            exclude_prefix: list, optional (default=['Mt-', 'Rpl', 'Rps'])
                Gene name prefixes to exclude.

        Returns
        -------            tuple
                (gene_labels, gene_indices):
                - gene_labels: Filtered gene names
                - gene_indices: Indices of kept genes

        Notes:
            - Removes low-expressed genes
            - Excludes common housekeeping genes
            - Useful for focusing on tissue-specific genes
            - Results can be used for downstream analysis
        """
        self.umi_thresh=umi_thresh
        from ..external.gaston.filter_genes import filter_genes
        if issparse(self.counts_mat_restrict):
            counts_mat_restrict=self.counts_mat_restrict.toarray()
        else:
            counts_mat_restrict=self.counts_mat_restrict
        idx_kept, gene_labels_idx = filter_genes(counts_mat_restrict, self.adata.var.index.to_numpy(), 
                                                umi_threshold = umi_thresh,exclude_prefix=exclude_prefix)
        self.gene_labels_idx=gene_labels_idx
        self.idx_kept=idx_kept
        return idx_kept, gene_labels_idx

    def pw_linear_fit(self,cell_type_df=None,ct_list=[],
                      isodepth_mult_factor=0.01, **kwargs):
        r"""Perform piecewise linear fitting of gene expression.
        
        This method fits piecewise linear functions to gene expression patterns
        along the spatial depth gradient.

        Parameters
        ----------            cell_type_df: pandas.DataFrame, optional (default=None)
                Cell type annotations if available.
            ct_list: list, optional (default=[])
                List of cell types to analyze.
            **kwargs:
                Additional arguments for piecewise fitting.

        Returns
        -------            dict
                Dictionary containing fitted parameters and statistics.

        Notes:
            - Models expression changes along spatial axis
            - Can incorporate cell type information
            - Identifies expression breakpoints
            - Useful for finding spatial transition points
        """
        from ..external.gaston.segmented_fit import pw_linear_fit
        if issparse(self.counts_mat_restrict):
            counts_mat_restrict=self.counts_mat_restrict.toarray()
        else:
            counts_mat_restrict=self.counts_mat_restrict
            
        pw_fit_dict=pw_linear_fit(counts_mat_restrict, self.gaston_labels_restrict, self.gaston_isodepth_restrict,
                                  cell_type_df, ct_list, idx_kept=self.idx_kept, umi_threshold=self.umi_thresh, 
                                  isodepth_mult_factor=isodepth_mult_factor, **kwargs)                
        self.pw_fit_dict=pw_fit_dict
        return pw_fit_dict

    def bin_data(self,cell_type_df=None,
                 num_bins=15,q_discont=0.95,q_cont=0.8,**kwargs):
        r"""Bin data along spatial depth gradient.
        
        This method bins spots and their expression data based on isodepth values,
        useful for analyzing trends along spatial axes.

        Parameters
        ----------            cell_type_df: pandas.DataFrame, optional (default=None)
                Cell type annotations if available.
            num_bins: int, optional (default=15)
                Number of spatial bins to create.
            q_discont: float, optional (default=0.95)
                Quantile threshold for discontinuous patterns.
            q_cont: float, optional (default=0.8)
                Quantile threshold for continuous patterns.
            **kwargs:
                Additional arguments for binning process.

        Returns
        -------            tuple
                (binned_data, bin_edges, statistics):
                - binned_data: Expression data per bin
                - bin_edges: Isodepth values defining bins
                - statistics: Summary statistics per bin

        Notes:
            - Creates equal-width bins along isodepth axis
            - Computes statistics within each bin
            - Can incorporate cell type information
            - Useful for trajectory analysis
        """
        from ..external.gaston.binning_and_plotting import bin_data
        from ..external.gaston.spatial_gene_classification import get_discont_genes, get_cont_genes
        if issparse(self.counts_mat_restrict):
            counts_mat_restrict=self.counts_mat_restrict.toarray()
        else:
            counts_mat_restrict=self.counts_mat_restrict
            
        binning_output=bin_data(counts_mat_restrict, 
                                self.gaston_labels_restrict, 
                                self.gaston_isodepth_restrict, 
                                cell_type_df, self.adata.var.index.to_numpy(), 
                                idx_kept=self.idx_kept, num_bins=num_bins, umi_threshold=self.umi_thresh,
                                **kwargs)
        self.binning_output=binning_output
        self.discont_genes_layer=get_discont_genes(self.pw_fit_dict, binning_output,q=q_discont)
        self.cont_genes_layer=get_cont_genes(self.pw_fit_dict, binning_output,q=q_cont) 
        return binning_output

    def get_restricted_adata(self,offset=10**6,):
        r"""Create AnnData object from restricted data.
        
        This method constructs a new AnnData object containing only the spots
        within the restricted isodepth range.

        Parameters
        ----------            offset: float, optional (default=10**6)
                Offset for coordinate normalization.

        Returns
        -------            anndata.AnnData
                AnnData object containing restricted data.

        Notes:
            - Preserves original data structure
            - Contains only filtered spots
            - Maintains spatial coordinates
            - Useful for downstream analysis
        """
        adata=self.adata
        # get restricted adata subset
        adata2=adata[self.locs, self.adata.var_names[self.idx_kept]]
        #adata2.obsm['spatial']=self.coords_mat_restrict
        adata2=adata2[:,self.gene_labels_idx]
        adata2.uns['gaston']={}
        adata2.uns['gaston']['isodepth']=self.gaston_isodepth_restrict
        adata2.uns['gaston']['labels']=self.gaston_labels_restrict

        slope_mat, intercept_mat, _, _ = self.pw_fit_dict['all_cell_types']

        gene_list = list(self.binning_output['gene_labels_idx']) # 获取基因列表
        
        # Filter gene_list to only include genes that are actually in adata2 and gene_labels_idx
        valid_genes = []
        for gene_name in gene_list:
            if gene_name in adata2.var_names and gene_name in self.gene_labels_idx:
                valid_genes.append(gene_name)
        
        # Update adata2 to only include valid genes
        adata2=adata2[:,valid_genes]
        all_gene_outputs = []

        for gene_name in tqdm(valid_genes):
            gene_index = np.where(self.gene_labels_idx == gene_name)[0]
            
            if len(gene_index) == 0:
                # This shouldn't happen with our validation above, but keep as safety check
                continue
                
            outputs = np.zeros(self.gaston_isodepth_restrict.shape[0])
            for i in range(self.gaston_isodepth_restrict.shape[0]):
                dom = int(self.gaston_labels_restrict[i])
                slope = slope_mat[gene_index, dom]
                intercept = intercept_mat[gene_index, dom]
                outputs[i] = np.log(offset) + intercept + slope * self.gaston_isodepth_restrict[i]

            all_gene_outputs.append(outputs)

        # Final validation
        if len(all_gene_outputs) == 0:
            raise ValueError("No valid gene outputs generated. Check gene filtering and binning steps.")
            
        if len(all_gene_outputs) != adata2.n_vars:
            raise ValueError(f"Gene output count ({len(all_gene_outputs)}) doesn't match adata2 gene count ({adata2.n_vars}). "
                           f"Valid genes found: {len(valid_genes)}")
            
        sparse_output_matrix = csr_matrix(all_gene_outputs)
        adata2.layers['GASTON_ReX']=sparse_output_matrix.T
        return adata2

    def plot_gene_pwlinear(self,gene,domain_colors,offset=10**6,
                           cell_type_list=None,pt_size=50,linear_fit=True,
                           ticksize=15, figsize=(4,2.5),
                           lw=3,domain_boundary_plotting=True):
        r"""Plot piecewise linear fit of gene expression.
        
        This method visualizes gene expression patterns and their piecewise
        linear fits along the spatial depth gradient.

        Parameters
        ----------            gene: str
                Name of gene to plot.
            domain_colors: dict or list
                Colors for each spatial domain.
            offset: float, optional (default=10**6)
                Offset for coordinate normalization.
            cell_type_list: list, optional (default=None)
                List of cell types to include.
            pt_size: float, optional (default=50)
                Size of scatter points.
            linear_fit: bool, optional (default=True)
                Whether to show linear fit lines.
            ticksize: float, optional (default=15)
                Size of axis tick labels.
            figsize: tuple, optional (default=(4,2.5))
                Figure size in inches.
            lw: float, optional (default=3)
                Line width for fits.
            domain_boundary_plotting: bool, optional (default=True)
                Whether to show domain boundaries.

        Returns
        -------            matplotlib.figure.Figure
                Figure containing the visualization.

        Notes:
            - Shows expression vs isodepth relationship
            - Highlights transition points
            - Can show multiple cell types
            - Useful for understanding spatial patterns
        """
        gene_name=gene
        print(f'gene {gene_name}: discontinuous jump after domain(s) {self.discont_genes_layer[gene_name]}') 
        print(f'gene {gene_name}: continuous gradient in domain(s) {self.cont_genes_layer[gene_name]}')

        # display log CPM (if you want to do CP500, set offset=500)
        #offset=10**6
        from ..external.gaston.binning_and_plotting import plot_gene_pwlinear
        plot_gene_pwlinear(gene_name, self.pw_fit_dict, self.gaston_labels_restrict, self.gaston_isodepth_restrict, 
                           self.binning_output, cell_type_list=cell_type_list, pt_size=pt_size, colors=domain_colors, 
                           linear_fit=linear_fit, ticksize=ticksize, figsize=figsize, offset=offset, lw=lw,
                           domain_boundary_plotting=domain_boundary_plotting)

    def plot_gene_raw(self,gene_name,rotate_angle=-90,
                      vmin=5,figsize=(6,3),s=10,**kwargs):
        r"""Plot raw gene expression in spatial coordinates.
        
        This method creates a spatial visualization of raw gene expression levels
        across the tissue section.

        Parameters
        ----------            gene_name: str
                Name of gene to plot.
            rotate_angle: float, optional (default=-90)
                Rotation angle in degrees for plot orientation.
            vmin: float, optional (default=5)
                Minimum value for color scale.
            figsize: tuple, optional (default=(6,3))
                Figure size in inches.
            s: float, optional (default=10)
                Size of scatter points.
            **kwargs:
                Additional arguments passed to plotting functions.

        Returns
        -------            matplotlib.figure.Figure
                Figure containing the expression visualization.

        Notes:
            - Shows spatial distribution of expression
            - Uses continuous color scale
            - Maintains tissue orientation
            - Useful for examining expression patterns
        """
        rotate=np.radians(rotate_angle)
        from ..external.gaston.binning_and_plotting import plot_gene_raw
        if issparse(self.counts_mat_restrict):
            counts_mat_restrict=self.counts_mat_restrict.toarray()
        else:
            counts_mat_restrict=self.counts_mat_restrict
        plot_gene_raw(gene_name, self.gene_labels_idx, counts_mat_restrict[:,self.idx_kept], 
                      self.S_restrict, vmin=vmin, figsize=figsize,s=s,rotate=rotate,**kwargs)
        plt.title(f'{gene_name} Raw Expression')

    def plot_gene_gastonrex(self,gene_name,rotate_angle=-90,
                            figsize=(6,3),s=10,**kwargs):
        r"""Plot gene expression with GASTON-specific visualization.
        
        This method creates a specialized visualization of gene expression using
        GASTON's representation enhancement.

        Parameters
        ----------            gene_name: str
                Name of gene to plot.
            rotate_angle: float, optional (default=-90)
                Rotation angle in degrees for plot orientation.
            figsize: tuple, optional (default=(6,3))
                Figure size in inches.
            s: float, optional (default=10)
                Size of scatter points.
            **kwargs:
                Additional arguments passed to plotting functions.

        Returns
        -------            matplotlib.figure.Figure
                Figure containing the enhanced visualization.

        Notes:
            - Uses GASTON-specific representation
            - Enhances spatial patterns
            - Maintains tissue orientation
            - Useful for detailed pattern analysis
        """
        rotate=np.radians(rotate_angle)
        from ..external.gaston.binning_and_plotting import plot_gene_function

        plot_gene_function(gene_name, self.S_restrict, self.pw_fit_dict, 
                           self.gaston_labels_restrict, self.gaston_isodepth_restrict, 
                           self.binning_output, figsize=figsize, s=s, rotate=rotate, **kwargs)
        plt.title(f'{gene_name} GASTON ReX')
