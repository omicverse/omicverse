"""Module providing a encapsulation of pySTAGATE."""
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import scanpy as sc
from anndata import AnnData
import numpy as np
import os
import warnings
from scipy.sparse import csr_matrix
from .._settings import add_reference
from .._registry import register_function
from .._settings import Colors


def _get_stagate_backend():
    from ..external.STAGATE_pyG import (
        Batch_Data,
        Cal_Spatial_Net,
        STAGATE,
        Stats_Spatial_Net,
        Transfer_pytorch_Data,
    )

    return Batch_Data, Cal_Spatial_Net, Transfer_pytorch_Data, Stats_Spatial_Net, STAGATE


@register_function(
    aliases=["STAGATE空间聚类", "pySTAGATE", "STAGATE", "空间聚类模型", "图注意力自编码器"],
    category="space",
    description="PyTorch implementation of STAGATE for spatial transcriptomics analysis using graph attention autoencoder",
    prerequisites={
        'optional_functions': []
    },
    requires={
        'obs': []  # Requires spatial coordinates in obs (user-specified spatial_key)
    },
    produces={
        'obsm': ['STAGATE'],
        'layers': ['STAGATE_ReX']
    },
    auto_fix='none',
    examples=[
        "# Basic STAGATE analysis",
        "stagate = ov.space.pySTAGATE(adata, num_batch_x=3, num_batch_y=2,",
        "                            spatial_key=['X','Y'], rad_cutoff=200)",
        "stagate.train()",
        "stagate.predicted()",
        "# Custom parameters",
        "stagate = ov.space.pySTAGATE(adata, num_batch_x=4, num_batch_y=3,",
        "                            num_epoch=1500, lr=0.001, device='cuda:0')",
        "# With pseudotime analysis",
        "stagate.cal_pSM(n_neighbors=20, resolution=1)",
        "# Access results",
        "embedding = adata.obsm['STAGATE']",
        "denoised_expr = adata.layers['STAGATE_ReX']"
    ],
    related=["space.svg", "space.clusters", "utils.cluster", "utils.refine_label"]
)
class pySTAGATE:
    """
    A class representing the PyTorch implementation of STAGATE (Spatial Transcriptomics Analysis using Graph Attention autoEncoder).

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData with coordinates and expression matrix.
    num_batch_x : int
        Number of tiles along x-axis for mini-batch graph construction.
    num_batch_y : int
        Number of tiles along y-axis for mini-batch graph construction.
    spatial_key : list, default=['X', 'Y']
        Coordinate columns in ``adata.obs`` used to build spatial graph.
    batch_size : int, default=1
        Number of tiled graphs per optimization step.
    rad_cutoff : int, default=200
        Radius cutoff when constructing spatial neighbors.
    num_epoch : int, default=1000
        Number of training epochs.
    lr : float, default=0.001
        Learning rate for Adam optimizer.
    weight_decay : float, default=1e-4
        L2 regularization strength.
    hidden_dims : list, default=[512, 30]
        Hidden-layer sizes of STAGATE encoder.
    device : str, default='cuda:0'
        Device specifier; falls back to CPU when CUDA is unavailable.

    Attributes:
        device: torch.device
            Device where the model is running.
        loader: DataLoader
            PyTorch DataLoader for batch processing.
        model: STAGATE
            The STAGATE model instance.
        optimizer: torch.optim.Adam
            Adam optimizer for model training.
        adata: AnnData
            Input annotated data matrix.
        data: torch_geometric.data.Data
            PyTorch geometric data object.

    Notes:
        The STAGATE model is designed for analyzing spatial transcriptomics data by incorporating
        spatial information through a graph attention autoencoder architecture.

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> adata = sc.read_h5ad('spatial_data.h5ad')
        >>> stagate = ov.space.pySTAGATE(adata, num_batch_x=3, num_batch_y=2)
        >>> stagate.train()
        >>> stagate.predicted()
    """

    def __init__(self,
                 adata: AnnData,
                 num_batch_x,
                 num_batch_y,
                 spatial_key: list = None,
                 batch_size: int = 1,
                rad_cutoff: int = 200,
                num_epoch: int = 1000,
                lr: float = 0.001,
                weight_decay: float = 1e-4,
                hidden_dims: list = None,
                device: str = 'cuda:0')-> None:
        """Initialize STAGATE training components.

        Parameters
        ----------
        adata : AnnData
            Spatial AnnData for representation learning.
        num_batch_x : int
            Number of x-axis tiles for batch graph generation.
        num_batch_y : int
            Number of y-axis tiles for batch graph generation.
        spatial_key : list, default=['X', 'Y']
            Coordinate columns in ``adata.obs``.
        batch_size : int, default=1
            Number of tiled samples per gradient step.
        rad_cutoff : int, default=200
            Radius cutoff for neighborhood graph construction.
        num_epoch : int, default=1000
            Number of epochs for training.
        lr : float, default=0.001
            Learning rate for optimizer.
        weight_decay : float, default=1e-4
            Weight decay for optimizer.
        hidden_dims : list, default=[512, 30]
            Hidden dimensions for STAGATE network.
        device : str, default='cuda:0'
            Compute device string.
        """
        if (
            isinstance(num_epoch, bool)
            or not isinstance(num_epoch, (int, np.integer))
            or num_epoch < 1
        ):
            raise ValueError(
                "`num_epoch` must be a positive integer; zero epochs would "
                "return an untrained random STAGATE embedding."
            )
        source_adata = adata
        adata = adata.copy()
        if spatial_key is None:
            if all(column in adata.obs for column in ('X', 'Y')):
                spatial_columns = ['X', 'Y']
                if 'spatial' in adata.obsm:
                    warnings.warn(
                        "pySTAGATE found both legacy adata.obs['X'/'Y'] and "
                        "adata.obsm['spatial']; the omitted spatial_key currently "
                        "preserves the legacy obs columns. Pass "
                        "spatial_key='spatial' explicitly to use canonical "
                        "coordinates. The automatic legacy preference will be "
                        "removed in a future release.",
                        FutureWarning,
                        stacklevel=2,
                    )
            elif 'spatial' in adata.obsm:
                spatial_columns = ['__ov_stagate_x', '__ov_stagate_y']
                coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)
                adata.obs[spatial_columns[0]] = coords[:, 0]
                adata.obs[spatial_columns[1]] = coords[:, 1]
            else:
                raise KeyError(
                    "STAGATE needs either legacy adata.obs['X'/'Y'] columns or "
                    "adata.obsm['spatial']."
                )
        elif isinstance(spatial_key, str):
            if spatial_key not in adata.obsm:
                raise KeyError(f"spatial_key {spatial_key!r} was not found in adata.obsm.")
            spatial_columns = ['__ov_stagate_x', '__ov_stagate_y']
            coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
            adata.obs[spatial_columns[0]] = coords[:, 0]
            adata.obs[spatial_columns[1]] = coords[:, 1]
        else:
            spatial_columns = list(spatial_key)

        if len(spatial_columns) != 2 or any(column not in adata.obs for column in spatial_columns):
            raise KeyError(
                "STAGATE needs either `adata.obsm['spatial']` or two coordinate "
                f"columns in adata.obs; received {spatial_columns}."
            )
        selected_coords = adata.obs[spatial_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(selected_coords).all():
            raise ValueError("STAGATE spatial coordinates must be finite.")
        adata.obsm['spatial'] = selected_coords
        hidden_dims = [512, 30] if hidden_dims is None else list(hidden_dims)
        # Initialize device
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.device=device
        Batch_Data, Cal_Spatial_Net, Transfer_pytorch_Data, Stats_Spatial_Net, STAGATE = _get_stagate_backend()

        # Create batches
        batch_list = Batch_Data(adata, num_batch_x=num_batch_x, num_batch_y=num_batch_y,
                                    spatial_key=spatial_columns, plot_Stats=True)
        for temp_adata in batch_list:
            Cal_Spatial_Net(temp_adata, rad_cutoff=rad_cutoff)

        # Transfer to PyTorch data format
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        data_list = [Transfer_pytorch_Data(adata) for adata in batch_list]
        for temp in data_list:
            temp.to(device)

        Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)
        data = Transfer_pytorch_Data(adata)
        Stats_Spatial_Net(adata)
        source_adata.uns['Spatial_Net'] = adata.uns['Spatial_Net'].copy()

        # batch_size=1 or 2
        self.loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

        # hyper-parameters
        self.num_epoch = num_epoch
        self.lr=lr
        self.weight_decay=weight_decay
        self.hidden_dims = hidden_dims
        self.adata=source_adata
        self.data=data
        self._is_fitted = False

        # Model and optimizer
        self.model = STAGATE(hidden_dims = [data_list[0].x.shape[1]]+self.hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                            lr=self.lr,
                                            weight_decay=self.weight_decay)

    def train(self):
        """
        Train the STAGATE model using the configured parameters.

        This method performs training of the STAGATE model for the specified number of epochs.
        For each epoch, it processes batches of data, computes the loss using mean squared error,
        and updates the model parameters through backpropagation.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes:
            - The training progress is displayed using a progress bar.
            - Gradient clipping is applied with a maximum norm of 5.
            - The model is automatically moved to the specified device (GPU/CPU).
        """       
        for epoch in tqdm(range(1, self.num_epoch+1)):
            for batch in self.loader:
                self.model.train()
                self.optimizer.zero_grad()
                z, out = self.model(batch.x, batch.edge_index)
                loss = F.mse_loss(batch.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optimizer.step()
        # The total network
        self.data.to(self.device)
        self._is_fitted = True

    def predicted(self):
        """
        Generate STAGATE representations and reconstruction values for all cells.

        This method runs the trained model in evaluation mode to generate:
        1. STAGATE embeddings (stored in adata.obsm['STAGATE'])
        2. Reconstructed expression values (stored in adata.layers['STAGATE_ReX'])

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes:
            - Negative values in reconstructed expression are set to 0
            - Results are stored directly in the AnnData object:
                - STAGATE embeddings: adata.obsm['STAGATE']
                - Reconstructed expression: adata.layers['STAGATE_ReX']
        """
        if not self._is_fitted:
            raise RuntimeError("Run `train()` before `predicted()`.")
        self.model.eval()
        z, out = self.model(self.data.x, self.data.edge_index)

        stagate_rep = z.to('cpu').detach().numpy()
        self.adata.obsm['STAGATE'] = stagate_rep
        rex = out.to('cpu').detach().numpy()
        rex[rex<0] = 0
        self.adata.layers['STAGATE_ReX'] = rex

        print('The STAGATE representation values are stored in adata.obsm["STAGATE"].')
        print('The rex values are stored in adata.layers["STAGATE_ReX"].')

    def cal_pSM(self, n_neighbors:int=20, resolution:int=1,
                max_cell_for_subsampling:int=5000,
                psm_key:str='pSM_STAGATE'):
        """
        Calculate the pseudo-spatial map using diffusion pseudotime (DPT) algorithm.

        This method computes a pseudo-spatial map by:
        1. Constructing a kNN graph using STAGATE embeddings
        2. Performing UMAP and Leiden clustering
        3. Computing diffusion pseudotime
        4. Storing results in the AnnData object

        Parameters
        ----------
        n_neighbors : int, default=20
            Neighbors used to build kNN graph from STAGATE embedding.
        resolution : int, default=1
            Leiden clustering resolution before DPT.
        max_cell_for_subsampling : int, default=5000
            Maximum number of cells used in root-cell distance estimation.
        psm_key : str, default='pSM_STAGATE'
            Output column name for pseudo-spatial scores in ``adata.obs``.

        Returns
        -------
        numpy.ndarray
            Pseudo-spatial map values derived from diffusion pseudotime.

        Notes:
            - If number of cells exceeds max_cell_for_subsampling, random subsampling is performed
            - The root cell for pseudotime calculation is chosen as the cell with maximum
              total distance to all other cells
            - Results are stored in adata.obs[psm_key]
        """

        from scipy.spatial import distance_matrix
        import numpy as np

        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors,
               use_rep='STAGATE')
        sc.tl.umap(self.adata)
        sc.tl.leiden(self.adata, resolution=resolution)
        sc.tl.paga(self.adata)
       # max_cell_for_subsampling = max_cell_for_subsampling
        if self.adata.shape[0] < max_cell_for_subsampling:
            sub_adata_x = self.adata.obsm['STAGATE']
        else:
            indices = np.arange(self.adata.shape[0])
            selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
            sub_adata_x = self.adata[selected_ind, :].obsm['STAGATE']

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

        psm_values = self.adata.obs[psm_key].to_numpy()
        return psm_values
        # End-of-file (EOF)


@register_function(
    aliases=["空间聚类分析", "clusters", "spatial_clustering", "多方法聚类", "空间域聚类"],
    category="space",
    description="Compute spatial representations with STAGATE, GraphST, CAST, BINARY, or BANKSY",
    prerequisites={
        'optional_functions': []
    },
    requires={
        'obs': []  # Requires spatial coordinates (user-specified)
    },
    produces={
        'obsm': [],  # Dynamic: depends on method (STAGATE, GraphST_embedding, CAST, BINARY)
        'uns': []    # Dynamic: may produce Spatial_Graph
    },
    auto_fix='none',
    examples=[
        "# Multiple clustering methods",
        "methods = ['STAGATE', 'GraphST']",
        "methods_kwargs = {",
        "    'STAGATE': {'num_batch_x': 3, 'num_batch_y': 2, 'device': 'cuda:0'},",
        "    'GraphST': {'device': 'cuda:0', 'n_pcs': 30}",
        "}",
        "adata = ov.space.clusters(adata, methods=methods,",
        "                         methods_kwargs=methods_kwargs)",
        "# Single method clustering",
        "methods_kwargs = {'BINARY': {'device': 'cuda:0', 'n_epochs': 1000}}",
        "adata = ov.space.clusters(adata, methods=['BINARY'],",
        "                         methods_kwargs=methods_kwargs)",
        "# Access results",
        "stagate_embedding = adata.obsm['STAGATE']",
        "graphst_embedding = adata.obsm['GraphST_embedding']"
    ],
    related=["space.pySTAGATE", "space.svg", "utils.cluster", "utils.refine_label"]
)
def clusters(adata,
             methods,
             methods_kwargs=None,
             batch_key=None,
             spatial_key='spatial',
             lognorm=50*1e4,
             ):
    """
    Perform clustering analysis on spatial transcriptomics data using multiple methods.

    This function supports multiple clustering methods including STAGATE, GraphST, CAST, and BINARY.
    Each method processes the spatial data differently and stores its results in the AnnData object.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData object to be clustered/embedded.
    methods : list
        Ordered list of methods to run, e.g. ``['STAGATE', 'GraphST']``.
    methods_kwargs : dict
        Method-specific configuration dictionary.
    batch_key : str, optional
        Batch/sample key in ``adata.obs`` for multi-sample methods.
    spatial_key : str, default='spatial'
        Spatial coordinate key in ``adata.obsm``.
    lognorm : float, default=50*1e4
        Target-sum scale used when recovering counts from log-normalized data.

    Returns
    -------
    AnnData
        Updated AnnData containing method-specific embeddings/layers.

    Notes:
        - For STAGATE: If n_top_genes is specified, highly variable genes are selected
        - For GraphST: Requires counts matrix in adata.layers['counts']
        - For CAST: Requires spatial coordinates in adata.obsm['spatial']
        - For BINARY: Can handle both raw counts and normalized data

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> adata = sc.read_h5ad('spatial_data.h5ad')
        >>> methods = ['STAGATE', 'GraphST']
        >>> methods_kwargs = {
        ...     'STAGATE': {'num_batch_x': 3, 'num_batch_y': 2},
        ...     'GraphST': {'device': 'cuda:0', 'n_pcs': 30}
        ... }
        >>> adata = ov.space.clusters(adata, methods, methods_kwargs)
    """
    import copy
    from scipy.sparse import issparse

    canonical_methods = {
        'stagate': 'STAGATE',
        'graphst': 'GraphST',
        'cast': 'CAST',
        'binary': 'BINARY',
        'banksy': 'Banksy',
    }
    if isinstance(methods, str):
        methods = [methods]
    methods = list(methods)
    unknown = [method for method in methods if str(method).lower() not in canonical_methods]
    if unknown:
        raise ValueError(
            f"Unsupported spatial clustering method(s): {unknown}. "
            f"Choose from: {list(canonical_methods.values())}."
        )
    methods = [canonical_methods[str(method).lower()] for method in methods]

    default_method_kwargs = {
        'STAGATE': {
            'num_batch_x': 3,
            'num_batch_y': 2,
            'spatial_key': ['X', 'Y'],
            'rad_cutoff': 200,
            'num_epoch': 1000,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'hidden_dims': [512, 30],
            'device': 'cuda:0',
            'n_top_genes': 2000,
        },
        'GraphST': {'device': 'cuda:0', 'n_pcs': 30},
        'CAST': {
            'output_path_t': 'result/CAST_gas/output',
            'device': 'cuda:0',
            'gpu_t': 0,
        },
        'BINARY': {
            'use_method': 'KNN',
            'cutoff': 6,
            'obs_key': 'BINARY_sample',
            'use_list': None,
            'pos_weight': 10,
            'device': 'cuda:0',
            'hidden_dims': [512, 30],
            'n_epochs': 1000,
            'lr': 0.001,
            'key_added': 'BINARY',
            'gradient_clipping': 5,
            'weight_decay': 0.0001,
            'verbose': True,
            'random_seed': 0,
            'n_top_genes': 2000,
        },
        'Banksy': {
            'num_neighbours': 15,
            'nbr_weight_decay': 'scaled_gaussian',
            'max_m': 1,
            'lambda_list': [0.2],
            'resolutions': [0.8],
            'add_nonspatial': False,
            'variance_balance': False,
            'match_labels': False,
            'filepath': 'result/Banksy',
        },
    }
    user_kwargs = methods_kwargs or {}
    normalized_user_kwargs = {}
    for method, params in user_kwargs.items():
        canonical = canonical_methods.get(str(method).lower(), str(method))
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise TypeError(f"methods_kwargs[{method!r}] must be a dict or None.")
        normalized_user_kwargs[canonical] = copy.deepcopy(params)
    methods_kwargs = {}
    for method in methods:
        methods_kwargs[method] = copy.deepcopy(default_method_kwargs[method])
        methods_kwargs[method].update(normalized_user_kwargs.get(method, {}))

    for method in methods:
        if method=='STAGATE':
            print('The STAGATE method is used to cluster the spatial data.')
            adata_copy=adata.copy()
            if issparse(adata_copy.obsm[spatial_key]):
                adata_copy.obsm[spatial_key]=adata_copy.obsm[spatial_key].toarray()
            adata_copy.obs['X']=adata_copy.obsm[spatial_key][:,0]
            adata_copy.obs['Y']=adata_copy.obsm[spatial_key][:,1]
            if 'STAGATE' not in methods_kwargs:
                methods_kwargs['STAGATE']={'num_batch_x':3,'num_batch_y':2,
                                           'spatial_key':['X','Y'],'rad_cutoff':200,'num_epoch':1000,'lr':0.001,
                                           'weight_decay':1e-4,'hidden_dims':[512, 30],'device':'cuda:0',
                                           'n_top_genes':2000}
            if 'n_top_genes' in methods_kwargs['STAGATE']:
                sc.pp.highly_variable_genes(adata_copy, flavor="seurat_v3", 
                                            n_top_genes=methods_kwargs['STAGATE']['n_top_genes'])
                del methods_kwargs['STAGATE']['n_top_genes']

            STA_obj=pySTAGATE(adata_copy,**methods_kwargs['STAGATE'])
            STA_obj.train()
            STA_obj.predicted()
            adata.obsm['STAGATE']=adata_copy.obsm['STAGATE']
            adata.layers['STAGATE_ReX']=adata_copy.layers['STAGATE_ReX']
            print(f'The STAGATE embedding are stored in adata.obsm["STAGATE"].\nShape: {adata.obsm["STAGATE"].shape}')
            add_reference(adata,'STAGATE','clustering with STAGATE')

        elif method=='GraphST':
            print('The GraphST method is used to embed the spatial data.')
            from ..external.GraphST import GraphST

            adata_copy=adata.copy()
            if 'GraphST' not in methods_kwargs:
                methods_kwargs['GraphST']={'device':'cuda:0','n_pcs':30,
                                           }
            if 'counts' not in adata_copy.layers:
                from ..pp import recover_counts
                if issparse(adata_copy.X):
                    pass 
                else:
                    adata_copy.X=csr_matrix(adata_copy.X)
                print('Recover the counts matrix from log-normalized data.')
                X_counts_recovered, size_factors_sub=recover_counts(adata_copy.X, lognorm, 
                                                                    lognorm*10,  log_base=None, 
                                                          chunk_size=10000)
                adata_copy.X=X_counts_recovered
            else:
                adata_copy.X=adata_copy.layers['counts']
            
            GRT_obj=GraphST(adata_copy, **methods_kwargs['GraphST'])
            adata_copy=GRT_obj.train()
            adata.obsm['GraphST_embedding']=adata_copy.obsm['GraphST_embedding']
            adata.obsm['graphst|original|X_pca']=adata_copy.obsm['graphst|original|X_pca']

            print(f'The GraphST embedding are stored in adata.obsm["GraphST_embedding"]. \nShape: {adata.obsm["GraphST_embedding"].shape}')
            add_reference(adata,'GraphST','clustering with GraphST')

        elif method=='CAST':
            print('The CAST method is used to embed the spatial data.')
            if issparse(adata.obsm[spatial_key]):
                adata.obsm[spatial_key]=adata.obsm[spatial_key].toarray()
            adata.obs['X']=adata.obsm[spatial_key][:,0]
            adata.obs['Y']=adata.obsm[spatial_key][:,1]

            if batch_key is None:
                adata.obs['CAST_sample']='sample1'
            else:
                adata.obs['CAST_sample']=adata.obs[batch_key]
            
            if 'CAST' not in methods_kwargs:
                methods_kwargs['CAST']={'output_path_t':'result/CAST_gas/output',
                                        'device':'cuda:0',
                                        'gpu_t':0}
            # Get the coordinates and expression data for each sample
            samples = np.unique(adata.obs['CAST_sample']) # used samples in adata
            coords_raw = {sample_t: np.array(adata.obs[['X','Y']])[adata.obs['CAST_sample'] == sample_t] for sample_t in samples}
            
            if ('norm_1e4' not in adata.layers) and ('counts' in adata.layers):
                normalized = sc.pp.normalize_total(
                    adata,
                    target_sum=1e4,
                    layer='counts',
                    inplace=False,
                )['X']
                adata.layers['norm_1e4'] = (
                    normalized.toarray() if issparse(normalized) else np.asarray(normalized)
                )
                exp_dict = {sample_t: adata[adata.obs['CAST_sample'] == sample_t].layers['norm_1e4'] for sample_t in samples}
            else:
                exp_dict = {sample_t: adata[adata.obs['CAST_sample'] == sample_t].X for sample_t in samples}
            
            output_path = methods_kwargs['CAST']['output_path_t']
            os.makedirs(output_path, exist_ok=True)

            from ..external.CAST import CAST_MARK
            embed_dict = CAST_MARK(coords_raw,exp_dict,**methods_kwargs['CAST'])

            from tqdm import tqdm
            adata.obsm['X_cast']=np.zeros((adata.shape[0],512))
            import pandas as pd
            adata.obsm['X_cast']=pd.DataFrame(adata.obsm['X_cast'],index=adata.obs.index)
            for key in tqdm(embed_dict.keys()):
                adata.obsm['X_cast'].loc[adata.obs['CAST_sample']==key]+=embed_dict[key].cpu().numpy()
            adata.obsm['X_cast']=adata.obsm['X_cast'].values
            print(f'The CAST embedding are stored in adata.obsm["X_cast"]. \nShape: {adata.obsm["X_cast"].shape}')
            add_reference(adata,'CAST','embedding with CAST')
        elif method=='BINARY':
            print('The BINARY method is used to embed the spatial data.')
            from ..external import BINARY
            if batch_key is None:
                adata.obs['BINARY_sample']='sample1'
            else:
                adata.obs['BINARY_sample']=adata.obs[batch_key]
            if 'BINARY' not in methods_kwargs:
                methods_kwargs['BINARY']={'use_method':'KNN',
                                          'cutoff':6,
                                          'obs_key':'BINARY_sample',
                                          'use_list':None,
                                          'pos_weight':10,
                                          'device':'cuda:0',
                                          'hidden_dims':[512, 30],
                                            'n_epochs': 1000,
                                            'lr':  0.001,
                                            'key_added': 'BINARY',
                                            'gradient_clipping': 5,
                                            'weight_decay': 0.0001,
                                            'verbose': True,
                                            'random_seed':0,
                                            #'lognorm':50*1e4,
                                            'n_top_genes':2000}
            adata_copy = BINARY.clean_adata(adata, save_obs=[methods_kwargs['BINARY']['obs_key']])
            if 'counts' in adata.layers:
                adata_copy.X=adata[adata_copy.obs.index].layers['counts']
            if 'counts' not in adata_copy.layers:
                from ..pp import recover_counts
                if issparse(adata_copy.X):
                    pass 
                else:
                    adata_copy.X=csr_matrix(adata_copy.X)
                print('Recover the counts matrix from log-normalized data.')
                X_counts_recovered, size_factors_sub=recover_counts(adata_copy.X, lognorm, 
                                                                    lognorm*10,  log_base=None, 
                                                          chunk_size=10000)
                adata_copy.X=X_counts_recovered
            else:
                adata_copy.X=adata_copy.layers['counts']
                #recover_counts(adata_copy.X, mult_value, max_range, log_base=None, chunk_size=1000)
            adata_copy = BINARY.Count2Binary(adata_copy)
            if 'n_top_genes' in methods_kwargs['BINARY']:
                sc.pp.highly_variable_genes(adata_copy, flavor="seurat_v3", n_top_genes=methods_kwargs['BINARY']['n_top_genes'])

            BINARY.Mutil_Construct_Spatial_Graph(adata_copy,
                                           use_method=methods_kwargs['BINARY']['use_method'],
                                           cutoff=methods_kwargs['BINARY']['cutoff'],
                                           obs_key=methods_kwargs['BINARY']['obs_key'],
                                           use_list=methods_kwargs['BINARY']['use_list'],
                                     )
            adata_copy = BINARY.train_BINARY(adata_copy,pos_weight=methods_kwargs['BINARY']['pos_weight'],
                                            device=methods_kwargs['BINARY']['device'],
                                            hidden_dims=methods_kwargs['BINARY']['hidden_dims'],
                                            n_epochs=methods_kwargs['BINARY']['n_epochs'],
                                            lr=methods_kwargs['BINARY']['lr'],
                                            key_added=methods_kwargs['BINARY']['key_added'],
                                            gradient_clipping=methods_kwargs['BINARY']['gradient_clipping'],
                                            weight_decay=methods_kwargs['BINARY']['weight_decay'],
                                            verbose=methods_kwargs['BINARY']['verbose'],
                                            random_seed=methods_kwargs['BINARY']['random_seed'])
            adata.obsm['BINARY']=adata_copy.obsm['BINARY']
            adata.uns['Spatial_Graph']=adata_copy.uns['Spatial_Graph']
            print(f'The binary embedding are stored in adata.obsm["BINARY"]. \nShape: {adata.obsm["BINARY"].shape}')
            add_reference(adata,'BINARY','clustering with BINARY')
        elif method=='Banksy' or method=='banksy':
            try: 
                import banksy
            except:
                print(f"{Colors.WARNING}banksy not installed. Install with: pip install pybanksy.{Colors.ENDC}")
                raise ValueError("banksy not installed.")
            from banksy.initialize_banksy import initialize_banksy
            from banksy.run_banksy import run_banksy_multiparam
            from ..pl import palette_112

            # Initialize BANKSY
            coord_keys = ('X', 'Y', spatial_key)  # Adjust based on your data
            adata.obs['X']=adata.obsm[spatial_key][:,0]
            adata.obs['Y']=adata.obsm[spatial_key][:,1]
            banksy_dict = initialize_banksy(
                adata,
                coord_keys=coord_keys,
                num_neighbours=methods_kwargs['Banksy']['num_neighbours'],
                nbr_weight_decay=methods_kwargs['Banksy']['nbr_weight_decay'],
                max_m=methods_kwargs['Banksy']['max_m'],
            )

            # Patch leidenalg Label repr to prevent Jupyter JSON serialization errors
            try:
                from banksy.labels import Label
                _orig_repr = Label.__repr__
                Label.__repr__ = lambda self: f'Label(n={getattr(self, "num_labels", "?")})'
            except (ImportError, AttributeError):
                _orig_repr = None

            try:
                results_df = run_banksy_multiparam(
                    adata,
                    banksy_dict,
                    lambda_list=methods_kwargs['Banksy']['lambda_list'],
                    resolutions=methods_kwargs['Banksy']['resolutions'],
                    color_list=palette_112,
                    annotation_key=None,
                    max_m=methods_kwargs['Banksy']['max_m'],
                    filepath=methods_kwargs['Banksy']['filepath'],
                    key=coord_keys,
                    #annotation_key='banksy_label',
                    add_nonspatial=methods_kwargs['Banksy']['add_nonspatial'],
                    variance_balance=methods_kwargs['Banksy']['variance_balance'],
                    match_labels=methods_kwargs['Banksy']['match_labels'],
                )
            finally:
                if _orig_repr is not None:
                    Label.__repr__ = _orig_repr
            result_keys=[i for i in results_df['adata'].keys()]
            for key in result_keys:
                adata.obsm[f'X_banksy_{key}']=results_df['adata'][key].obsm['reduced_pc_20']
            del results_df  # Prevent leidenalg Label objects from leaking into Jupyter

            add_reference(adata,'Banksy','clustering with Banksy')

        else:  # guarded above; retained as a defensive invariant
            raise RuntimeError(f"Unhandled spatial clustering method {method!r}.")
    return adata

@register_function(
    aliases=["合并类群", "merge_cluster", "cluster_merge", "类群合并", "合并空间类群"],
    category="space",
    description="Merge spatial clusters based on hierarchical clustering of their representation",
    prerequisites={
        'functions': []  # Requires prior clustering but method is flexible
    },
    requires={
        'obs': [],    # Dynamic: requires groupby column (user-specified)
        'obsm': []    # Dynamic: requires use_rep (user-specified)
    },
    produces={
        'obs': []  # Dynamic: creates {groupby}_tree column
    },
    auto_fix='escalate',
    examples=[
        "# Basic cluster merging",
        "result = ov.space.merge_cluster(adata, groupby='mclust_GraphST',",
        "                                use_rep='graphst|original|X_pca',",
        "                                threshold=0.2, plot=True)",
        "# STAGATE cluster merging",
        "result = ov.space.merge_cluster(adata, groupby='mclust_STAGATE',",
        "                                use_rep='STAGATE', threshold=0.05)",
        "# Custom merging parameters",
        "result = ov.space.merge_cluster(adata, groupby='leiden',",
        "                                use_rep='X_pca', threshold=0.1,",
        "                                plot=False)",
        "# Access merged clusters",
        "merged_labels = adata.obs[f'{groupby}_tree']"
    ],
    related=["space.clusters", "utils.cluster", "utils.refine_label"]
)
def merge_cluster(adata,
                  groupby='mclust',
                  use_rep='STAGATE',
                  threshold=0.05,
                  plot=True,
                  start_idx=0,
                  **kwargs):
    """
    Merge clusters based on hierarchical clustering of their representation.

    This function performs hierarchical clustering on existing clusters and merges them
    based on a distance threshold. It can optionally visualize the dendrogram showing
    the merging process.

    Parameters
    ----------
    adata : AnnData
        AnnData containing existing cluster labels and embeddings.
    groupby : str, default='mclust'
        Cluster label column in ``adata.obs`` to merge.
    use_rep : str, default='STAGATE'
        Embedding key in ``adata.obsm`` used for dendrogram distance.
    threshold : float, default=0.05
        Hierarchical-clustering distance threshold for merging.
    plot : bool, default=True
        Whether to display dendrogram with threshold line.
    start_idx : int, default=0
        Retained for backward-compatible calls. Actual category labels are now
        mapped directly, so this offset is no longer needed.
    **kwargs
        Extra arguments passed to ``scanpy.pl.dendrogram``.

    Returns
    -------
    dict
        Mapping from original cluster id to merged cluster label.

    Notes:
        - The function uses scipy's hierarchical clustering implementation
        - Merged clusters are stored in adata.obs[f'{groupby}_tree']
        - The dendrogram is stored in adata.uns[f'dendrogram_{groupby}']
        - Cluster labels in the output are prefixed with 'c'

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> adata = sc.read_h5ad('clustered_data.h5ad')
        >>> # Merge clusters using STAGATE representation
        >>> cluster_map = ov.space.merge_cluster(adata, 
        ...                                      groupby='leiden',
        ...                                      use_rep='STAGATE',
        ...                                      threshold=0.1)
        >>> # Access merged clusters
        >>> print(adata.obs['leiden_tree'])
    """
    import pandas as pd

    if groupby not in adata.obs:
        raise KeyError(f"{groupby!r} was not found in adata.obs.")
    if use_rep not in adata.obsm:
        raise KeyError(f"{use_rep!r} was not found in adata.obsm.")
    if start_idx != 0:
        import warnings

        warnings.warn(
            "`start_idx` is deprecated and ignored because merge_cluster now "
            "maps the actual category labels directly.",
            FutureWarning,
            stacklevel=2,
        )

    if not isinstance(adata.obs[groupby].dtype, pd.CategoricalDtype):
        adata.obs[groupby] = pd.Categorical(adata.obs[groupby])
    categories = list(adata.obs[groupby].cat.categories)

    sc.tl.dendrogram(adata,groupby=groupby,use_rep=use_rep)
    import numpy as np
    from scipy.cluster.hierarchy import fcluster

    # 你的链接矩阵
    linkage_matrix = adata.uns[f'dendrogram_{groupby}']['linkage']

    # 选择一个阈值来确定簇
    #threshold = 0.05  # 这个值需要根据具体情况调整

    # 使用fcluster来合并类别
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    
    # 创建字典
    cluster_dict = {}
    if len(clusters) != len(categories):
        raise RuntimeError(
            "Dendrogram cluster assignments do not match the number of input categories."
        )
    for category, cluster_id in zip(categories, clusters):
        key = f'c{cluster_id}'
        if key not in cluster_dict:
            cluster_dict[key] = []
        cluster_dict[key].append(category)
    
    reversed_dict = {}
    for key, values in cluster_dict.items():
        for value in values:
            reversed_dict[str(value)] = key
    
    #adata.obs['mclust_tree']=adata.obs['mclust'].map(reversed_dict)
    merged = adata.obs[groupby].astype(str).map(reversed_dict)
    adata.obs[f'{groupby}_tree'] = pd.Categorical(merged)
    print(f'The merged cluster information is stored in adata.obs["{groupby}_tree"].')
    if plot:
        ax=sc.pl.dendrogram(adata,groupby=groupby,show=False,**kwargs)
        ax.plot((ax.get_xticks().min(),ax.get_xticks().max()),(threshold,threshold))
    return reversed_dict
