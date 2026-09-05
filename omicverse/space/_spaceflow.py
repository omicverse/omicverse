r"""Module providing encapsulation of SpaceFlow for spatial flow analysis."""
import copy
import random
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from tqdm import tqdm
import torch.nn as nn

import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from sklearn.neighbors import kneighbors_graph
from .._settings import add_reference
from .._registry import register_function

sf_install = False


def _sampled_regularization_distances(z, coords, edge_subset_sz):
    """Sample independent cell pairs and return normalized latent/spatial distances."""
    subset_1 = torch.randint(0, z.shape[0], (edge_subset_sz,), device=z.device)
    subset_2 = torch.randint(0, z.shape[0], (edge_subset_sz,), device=z.device)
    pdist = torch.nn.PairwiseDistance(p=2, eps=0)
    z_dists = pdist(
        torch.index_select(z, 0, subset_1),
        torch.index_select(z, 0, subset_2),
    )
    spatial_dists = pdist(
        torch.index_select(coords, 0, subset_1),
        torch.index_select(coords, 0, subset_2),
    )
    z_dists = z_dists / torch.clamp(
        torch.max(z_dists), min=torch.finfo(z_dists.dtype).eps
    )
    spatial_dists = spatial_dists / torch.clamp(
        torch.max(spatial_dists), min=torch.finfo(spatial_dists.dtype).eps
    )
    return z_dists, spatial_dists


@register_function(
    aliases=["SpaceFlow空间流", "pySpaceFlow", "SpaceFlow", "空间流分析", "伪空间图谱"],
    category="space",
    description="SpaceFlow spatial flow analysis using deep graph neural networks with spatial regularization",
    prerequisites={
        'functions': [],
        'optional_functions': ['preprocess', 'scale']
    },
    requires={
        'obsm': ['spatial']
    },
    produces={
        'obsm': ['spaceflow'],
        'obs': ['pSM_spaceflow']
    },
    auto_fix='none',
    examples=[
        "# Basic SpaceFlow analysis",
        "spaceflow = ov.space.pySpaceFlow(adata)",
        "embedding = spaceflow.train(spatial_regularization_strength=0.1,",
        "                            z_dim=50, epochs=1000)",
        "# Calculate pseudo-spatial map",
        "spaceflow.cal_pSM(n_neighbors=20, resolution=1,",
        "                   psm_key='pSM_spaceflow')",
        "# Custom parameters",
        "embedding = spaceflow.train(spatial_regularization_strength=0.2,",
        "                            lr=1e-3, gpu=0, random_seed=42)",
        "# Access results",
        "spatial_embedding = adata.obsm['spaceflow']",
        "psm_values = adata.obs['pSM_spaceflow']"
    ],
    related=["space.svg", "space.pySTAGATE", "space.clusters"]
)
class pySpaceFlow(object):
    r"""SpaceFlow spatial flow analysis class.
    
    SpaceFlow is a deep learning method for analyzing spatial transcriptomics data
    by learning spatially-aware cell representations. It combines graph neural networks
    with spatial regularization to capture both transcriptional and spatial relationships
    between cells.

    The method:
    1. Constructs a spatial neighborhood graph
    2. Learns embeddings using deep graph infomax
    3. Applies spatial regularization to preserve spatial structure
    4. Generates pseudo-spatial maps for trajectory analysis

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData containing expression and coordinates in
        ``adata.obsm['spatial']``.

    Attributes:
        adata: AnnData
            Input annotated data matrix containing:
            - Gene expression data in adata.X
            - Spatial coordinates in adata.obsm['spatial']
        sf: SpaceFlow
            Internal SpaceFlow object for computations
        embedding: array
            Learned spatial-aware embeddings after training

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load spatial data
        >>> adata = sc.read_visium(...)
        >>> # Initialize SpaceFlow
        >>> spaceflow = ov.space.pySpaceFlow(adata)
        >>> # Train model
        >>> embedding = spaceflow.train(
        ...     spatial_regularization_strength=0.1,
        ...     z_dim=50,
        ...     epochs=1000
        ... )
        >>> # Calculate pseudo-spatial map
        >>> psm = spaceflow.cal_pSM(n_neighbors=20)
    """
    def __init__(self,adata) -> None:
        r"""Initialize SpaceFlow spatial analysis object.
        
        Parameters
        ----------
        adata : AnnData
            Spatial AnnData used for SpaceFlow embedding.

        Notes:
            - Automatically checks for SpaceFlow package installation
            - Constructs initial spatial neighborhood graph
            - Uses 10 nearest neighbors for graph construction
            - Stores SpaceFlow object in self.sf
        """
        global sf_install
        try:
            from ..external.spaceflow import SpaceFlow
            sf_install=True
        except ImportError as e:
            raise ImportError(
                'Please install the SpaceFlow: `pip install SpaceFlow`.'
            ) from e
        from ..external.spaceflow import SpaceFlow
        sf = SpaceFlow(adata=adata, 
                         spatial_locs=adata.obsm['spatial'])

        spatial_locs = adata.obsm['spatial']
        try:
            spatial_graph = sf.graph_alpha(spatial_locs, n_neighbors=10)
        except ImportError as exc:
            if getattr(exc, 'name', None) == 'gudhi' or 'gudhi' in str(exc):
                raise ImportError(
                    "pySpaceFlow requires Gudhi for spatial graph construction. "
                    "Install it with `pip install gudhi` "
                    "or `pip install gudhi`."
                ) from exc
            raise

        sf.adata_preprocessed = adata
        sf.spatial_graph = spatial_graph
        self.sf = sf
        self.adata = adata

    def train(self,spatial_regularization_strength=0.1, 
              z_dim=50, lr=1e-3, epochs=1000, max_patience=50, 
              min_stop=100, random_seed=42, gpu=0, 
              regularization_acceleration=True, edge_subset_sz=1000000):
        r"""Train SpaceFlow model for spatial embedding.
        
        This method trains a graph neural network to learn spatially-aware cell
        representations using deep graph infomax with spatial regularization.

        Parameters
        ----------
        spatial_regularization_strength : float, default=0.1
            Weight of spatial consistency regularization.
        z_dim : int, default=50
            Output latent embedding dimension.
        lr : float, default=1e-3
            Learning rate.
        epochs : int, default=1000
            Maximum training epochs.
        max_patience : int, default=50
            Patience for early stopping.
        min_stop : int, default=100
            Minimum epochs before early stopping can trigger.
        random_seed : int, default=42
            Random seed.
        gpu : int, default=0
            Preferred CUDA index.
        regularization_acceleration : bool, default=True
            Whether to subsample edges for regularization speedup.
        edge_subset_sz : int, default=1000000
            Number of sampled edges when acceleration is enabled.

        Returns
        -------
        numpy.ndarray
            Learned spatial embedding of shape ``(n_cells, z_dim)``.

        Notes:
            - Uses deep graph infomax for self-supervised learning
            - Applies spatial regularization to preserve spatial structure
            - Employs early stopping based on loss convergence
            - Supports GPU acceleration when available
            - Results are stored in adata.obsm['spaceflow']
            - Progress is shown with a progress bar
        """
        from ..external.spaceflow import sparse_mx_to_torch_edge_list, corruption

        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError('epochs must be a positive integer.')
        if edge_subset_sz < 1:
            raise ValueError('edge_subset_sz must be positive.')

        adata_preprocessed, spatial_graph = self.sf.adata_preprocessed, self.sf.spatial_graph
        if not adata_preprocessed:
            print("Data has not been preprocessed, please run preprocessing_data() method first!")
            return

        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
        model = DeepGraphInfomax(
            hidden_channels=z_dim, encoder=GraphEncoder(adata_preprocessed.shape[1], z_dim),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption).to(device)

        expr = adata_preprocessed.X.todense() if type(adata_preprocessed.X).__module__ != np.__name__ else adata_preprocessed.X
        expr = torch.tensor(expr).float().to(device)

        edge_list = sparse_mx_to_torch_edge_list(spatial_graph).to(device)

        model.train()
        min_loss = np.inf
        patience = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_params = copy.deepcopy(model.state_dict())

        for epoch in tqdm(range(epochs)):
            train_loss = 0.0
            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            z, neg_z, summary = model(expr, edge_list)
            loss = model.loss(z, neg_z, summary)

            coords = torch.tensor(adata_preprocessed.obsm['spatial']).float().to(device)
            if regularization_acceleration or adata_preprocessed.shape[0] > 5000:
                z_dists, sp_dists = _sampled_regularization_distances(
                    z,
                    coords,
                    edge_subset_sz,
                )
                n_items = z_dists.size(dim=0)
            else:
                z_dists = torch.cdist(z, z, p=2)
                z_dists = torch.div(
                    z_dists,
                    torch.clamp(torch.max(z_dists), min=torch.finfo(z_dists.dtype).eps),
                ).to(device)
                sp_dists = torch.cdist(coords, coords, p=2)
                sp_dists = torch.div(
                    sp_dists,
                    torch.clamp(torch.max(sp_dists), min=torch.finfo(sp_dists.dtype).eps),
                ).to(device)
                n_items = z.size(dim=0) * z.size(dim=0)

            penalty_1 = torch.div(torch.sum(torch.mul(1.0 - z_dists, sp_dists)), n_items).to(device)
            loss = loss + spatial_regularization_strength * penalty_1

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if train_loss > min_loss:
                patience += 1
            else:
                patience = 0
                min_loss = train_loss
                best_params = copy.deepcopy(model.state_dict())
            if patience > max_patience and epoch > min_stop:
                break

        model.load_state_dict(best_params)

        z, _, _ = model(expr, edge_list)
        embedding = z.cpu().detach().numpy()

        self.sf.embedding = embedding
        self.adata.obsm['spaceflow']=self.sf.embedding.copy()
        add_reference(self.adata,'SpaceFlow','embedding with SpaceFlow')

        return embedding

    def cal_pSM(self,n_neighbors:int=20,resolution:int=1,
                       max_cell_for_subsampling:int=5000,
                       psm_key='pSM_spaceflow'):
        r"""Calculate pseudo-spatial map using diffusion pseudotime.
        
        This method constructs a pseudo-spatial map by computing diffusion pseudotime
        on the learned embeddings, useful for analyzing spatial trajectories and
        organization patterns.

        Parameters
        ----------
        n_neighbors : int, default=20
            Neighbors used for graph construction on SpaceFlow embedding.
        resolution : int, default=1
            Leiden resolution before DPT.
        max_cell_for_subsampling : int, default=5000
            Max cells used in root-cell distance calculation.
        psm_key : str, default='pSM_spaceflow'
            Output column in ``adata.obs`` for pseudotime scores.

        Returns
        -------
        numpy.ndarray
            Pseudo-spatial ordering values.

        Notes:
            - Constructs neighborhood graph from embeddings
            - Performs UMAP and Leiden clustering
            - Uses PAGA for trajectory inference
            - Computes diffusion pseudotime
            - Results are stored in adata.obs[psm_key]
            - Useful for ordering cells along spatial axes
            - Handles large datasets through subsampling
        """
        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors,
               use_rep='spaceflow')
        sc.tl.umap(self.adata)
        sc.tl.leiden(self.adata, resolution=resolution)
        sc.tl.paga(self.adata)

        if self.adata.shape[0] < max_cell_for_subsampling:
            sub_adata_x = self.adata.obsm['spaceflow']
        else:
            indices = np.arange(self.adata.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = self.adata[selected_ind, :].obsm['spaceflow']

        sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
        root_in_subset = int(np.argmax(sum_dists))
        self.adata.uns['iroot'] = (
            root_in_subset
            if self.adata.shape[0] < max_cell_for_subsampling
            else int(selected_ind[root_in_subset])
        )
        sc.tl.diffmap(self.adata)
        sc.tl.dpt(self.adata)
        self.adata.obs.rename({"dpt_pseudotime": psm_key}, axis=1, inplace=True)
        print(f'The pseudo-spatial map values are stored in adata.obs["{psm_key}"].')
        add_reference(self.adata,'SpaceFlow','pseudo-spatial map with SpaceFlow')
        psm_values = self.adata.obs[psm_key].to_numpy()
        return psm_values


class GraphEncoder(nn.Module):
    r"""Graph convolutional encoder for SpaceFlow.
    
    This class implements a two-layer graph convolutional network (GCN) that serves
    as the encoder component in the SpaceFlow model. It learns to transform gene
    expression data into a spatially-aware latent representation.

    Architecture:
    1. Input layer: Gene expression features
    2. First GCN layer with PReLU activation
    3. Second GCN layer with PReLU activation
    4. Output layer: Learned embeddings

    Attributes:
        conv1: GCNConv
            First graph convolutional layer
        conv2: GCNConv
            Second graph convolutional layer
        prelu: PReLU
            Parametric ReLU activation function

    Notes:
        - Uses PyTorch Geometric's GCNConv implementation
        - Learns spatially-aware representations
        - Preserves both local and global structure
        - Part of the deep graph infomax framework
    """
    def __init__(self, in_channels, hidden_channels):
        super(GraphEncoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False)
        self.prelu = nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        self.prelu2 = nn.PReLU(hidden_channels)

    def forward(self, x, edge_index):
        r"""Forward propagation through graph encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Node-feature matrix.
        edge_index : torch.Tensor
            Sparse edge index in COO format.

        Returns
        -------
        torch.Tensor
            Encoded node embeddings.
        """
        x = self.conv(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x
    # End-of-file (EOF)
