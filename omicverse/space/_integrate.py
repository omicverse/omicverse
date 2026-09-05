"""Module for integrating spatial transcriptomics data across different conditions.

This module implements STAligner, a method for integrating spatial transcriptomics data
across different experimental conditions, technologies, and developmental stages. The
integration preserves both gene expression patterns and spatial organization.

Key features:
1. Spatial network construction
2. Graph neural network-based integration
3. Mutual nearest neighbor alignment
4. Batch effect correction
5. Cross-condition comparison

References:
    Zhou, X., Dong, K. & Zhang, S. Integrating spatial transcriptomics data 
    across different conditions, technologies and developmental stages. 
    Nat Comput Sci 3, 894–906 (2023)
"""
__author__ = "Xiang Zhou"
__email__ = "xzhou@amss.ac.cn"
__citation__ = "Zhou, X., Dong, K. & Zhang, S. Integrating spatial transcriptomics data across different conditions, technologies and developmental stages. Nat Comput Sci 3, 894–906 (2023)"

import numpy as np
import pandas as pd
from collections.abc import Mapping
from tqdm import tqdm
import scipy.sparse as sp
import sklearn.neighbors
import warnings

import torch

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .._settings import add_reference
from .._registry import register_function


def _get_staligner_backend():
    from ..external.STAligner.STALIGNER import STAligner
    from ..external.STAligner.mnn_utils import create_dictionary_mnn

    return STAligner, create_dictionary_mnn


def _obs_names_match_with_optional_suffix(source_names, combined_names):
    """Match source names to a combined block, allowing one uniform suffix."""
    source_names = np.asarray(source_names, dtype=str)
    combined_names = np.asarray(combined_names, dtype=str)
    if source_names.shape != combined_names.shape:
        return False
    if np.array_equal(source_names, combined_names):
        return True
    if not len(source_names):
        return True
    suffixes = [
        combined[len(source):] if combined.startswith(source) else None
        for source, combined in zip(source_names, combined_names)
    ]
    return (
        all(suffix is not None for suffix in suffixes)
        and len(set(suffixes)) == 1
        and bool(suffixes[0])
    )


def _validated_staligner_adjacency(batch, batch_index):
    """Return an adjacency aligned to current rows, preferring named edges."""
    if 'adj' not in batch.uns:
        raise KeyError(
            f"Batch_list[{batch_index}] needs uns['adj'] from Cal_Spatial_Net()."
        )
    stored = sp.csr_matrix(batch.uns['adj'])
    if stored.shape != (batch.n_obs, batch.n_obs):
        raise ValueError(
            f"Batch_list[{batch_index}].uns['adj'] has shape {stored.shape}; "
            f"expected ({batch.n_obs}, {batch.n_obs}). Rerun Cal_Spatial_Net()."
        )

    named_edges = batch.uns.get('Spatial_Net')
    if not isinstance(named_edges, pd.DataFrame) or not {
        'Cell1', 'Cell2'
    }.issubset(named_edges.columns):
        recorded_names = batch.uns.get('_omicverse_staligner_obs_names')
        if recorded_names is None:
            raise ValueError(
                f'Batch_list[{batch_index}] has only a positional adjacency without node identities. '
                'Rerun Cal_Spatial_Net() before training.'
            )
        if recorded_names is not None and not np.array_equal(
            np.asarray(recorded_names, dtype=str),
            np.asarray(batch.obs_names, dtype=str),
        ):
            raise ValueError(
                f"Batch_list[{batch_index}] rows changed after its positional "
                "adjacency was built. Rerun Cal_Spatial_Net()."
            )
        return stored

    name_to_pos = {name: i for i, name in enumerate(batch.obs_names)}
    row = named_edges['Cell1'].map(name_to_pos)
    col = named_edges['Cell2'].map(name_to_pos)
    if row.isna().any() or col.isna().any():
        raise ValueError(
            f"Batch_list[{batch_index}].uns['Spatial_Net'] references observations "
            "that are no longer present. Rerun Cal_Spatial_Net()."
        )
    named = sp.coo_matrix(
        (
            np.ones(len(named_edges), dtype=np.float64),
            (row.to_numpy(dtype=int), col.to_numpy(dtype=int)),
        ),
        shape=(batch.n_obs, batch.n_obs),
    ).tocsr()
    named = named + sp.eye(batch.n_obs, format='csr')
    named.data[:] = 1.0
    named.eliminate_zeros()

    stored_binary = stored.copy()
    stored_binary.data[:] = 1.0
    stored_binary.eliminate_zeros()
    if (stored_binary != named).nnz:
        warnings.warn(
            f"Batch_list[{batch_index}].uns['adj'] is not aligned with its named "
            "Spatial_Net edges; using the name-aligned graph. Rerun "
            "Cal_Spatial_Net() to refresh the stored adjacency.",
            UserWarning,
            stacklevel=3,
        )
    return named


@register_function(
    aliases=["空间网络构建", "Cal_Spatial_Net", "spatial_network", "空间邻域网络", "构建空间图"],
    category="space",
    description="Construct spatial neighbor networks for spatial transcriptomics integration",
    prerequisites={
        'optional_functions': []
    },
    requires={
        'obsm': ['spatial']  # Spatial coordinates required
    },
    produces={
        'uns': ['Spatial_Net', 'adj']
    },
    auto_fix='none',
    examples=[
        "# Radius-based spatial network",
        "ov.space.Cal_Spatial_Net(adata, rad_cutoff=150, model='Radius')",
        "# K-nearest neighbor network",
        "ov.space.Cal_Spatial_Net(adata, k_cutoff=6, model='KNN')",
        "# Custom parameters",
        "ov.space.Cal_Spatial_Net(adata, rad_cutoff=200, max_neigh=100,",
        "                         model='Radius', verbose=True)",
        "# Access network results",
        "spatial_graph = adata.uns['Spatial_Net']",
        "adjacency_matrix = adata.uns['adj']"
    ],
    related=["space.pySTAligner", "space.clusters", "space.pySTAGATE"]
)
def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='Radius', verbose=True):
    r"""Construct spatial neighbor networks for spatial integration.
    
    This function builds a spatial neighborhood graph by connecting spots based
    on their physical distances. It supports both radius-based and k-nearest
    neighbor approaches for network construction.

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData containing coordinates in ``obsm['spatial']``.
    rad_cutoff : float, optional
        Radius threshold when ``model='Radius'``.
    k_cutoff : int, optional
        Number of nearest neighbors when ``model='KNN'``.
    max_neigh : int, default=50
        Maximum neighbors queried before filtering by model.
    model : {'Radius', 'KNN'}, default='Radius'
        Strategy for building spatial edges.
    verbose : bool, default=True
        Whether to print graph statistics.

    Returns
    -------
    None
        Writes ``adata.uns['Spatial_Net']`` and ``adata.uns['adj']``.

    Notes:
        - For STAligner, adjust rad_cutoff to ensure 5-10 neighbors per spot
        - Includes self-loops in adjacency matrix
        - Uses ball_tree algorithm for efficient neighbor search
        - Memory efficient implementation for large datasets
        - Critical for downstream integration tasks
    """
    if model not in {'Radius', 'KNN'}:
        raise ValueError("`model` must be either 'Radius' or 'KNN'.")
    if adata.n_obs == 0:
        raise ValueError("Cannot construct a spatial graph for an empty AnnData object.")
    if not adata.obs_names.is_unique:
        raise ValueError('Spatial graph construction requires unique observation names.')
    if max_neigh < 1:
        raise ValueError("`max_neigh` must be a positive integer.")
    if model == 'Radius' and rad_cutoff is None:
        raise ValueError("`rad_cutoff` is required when model='Radius'.")
    if model == 'KNN' and (k_cutoff is None or k_cutoff < 1):
        raise ValueError("A positive `k_cutoff` is required when model='KNN'.")
    if 'spatial' not in adata.obsm:
        raise KeyError("adata.obsm['spatial'] is required to construct the graph.")
    spatial = np.asarray(adata.obsm['spatial'])
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must be an n_obs-by-2 coordinate array.")
    if not np.isfinite(spatial[:, :2]).all():
        raise ValueError("adata.obsm['spatial'] contains non-finite coordinates.")
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(spatial[:, :2])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    requested_neigh = int(max_neigh)
    if model == 'KNN':
        requested_neigh = max(requested_neigh, int(k_cutoff))
    query_neigh = min(requested_neigh + 1, adata.n_obs)
    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=query_neigh, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)

    if model == 'KNN':
        effective_k = min(int(k_cutoff), adata.n_obs - 1)
        if effective_k < int(k_cutoff):
            warnings.warn(
                f"`k_cutoff={k_cutoff}` exceeds the {adata.n_obs - 1} available "
                f"non-self neighbors; using {effective_k}.",
                UserWarning,
                stacklevel=2,
            )

    KNN_list = []
    for it in range(indices.shape[0]):
        # With duplicated coordinates sklearn may return a different zero-distance
        # spot before the query spot. Remove self by identity, never by position.
        nonself = indices[it] != it
        row_indices = indices[it][nonself]
        row_distances = distances[it][nonself]
        if model == 'KNN':
            row_indices = row_indices[:effective_k]
            row_distances = row_distances[:effective_k]
        KNN_list.append(
            pd.DataFrame(
                {
                    'Cell1': np.repeat(it, len(row_indices)),
                    'Cell2': row_indices,
                    'Distance': row_distances,
                }
            )
        )
    KNN_df = pd.concat(KNN_list, ignore_index=True) if KNN_list else pd.DataFrame(
        columns=['Cell1', 'Cell2', 'Distance']
    )

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    if verbose:
        print(f'The graph contains {Spatial_Net.shape[0]} edges, {adata.n_obs} cells.')
        print(f'{(Spatial_Net.shape[0] / adata.n_obs):.4f} neighbors per cell on average.')
    adata.uns['Spatial_Net'] = Spatial_Net

    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G
    adata.uns['_omicverse_staligner_obs_names'] = np.asarray(
        adata.obs_names,
        dtype=str,
    )


@register_function(
    aliases=["STAligner空间整合", "pySTAligner", "STAligner", "空间数据整合", "空间转录组整合"],
    category="space",
    description="STAligner for integrating spatial transcriptomics data across conditions and technologies",
    prerequisites={
        'functions': ['Cal_Spatial_Net'],
        'optional_functions': []
    },
    requires={
        'obsm': ['spatial'],
        'obs': [],  # Requires batch_name column (user-specified)
        'uns': ['Spatial_Net', 'adj']  # From Cal_Spatial_Net
    },
    produces={
        'obsm': ['STAligner', 'STAligner_embed']
    },
    auto_fix='auto',
    examples=[
        "# Basic STAligner integration",
        "staligner = ov.space.pySTAligner(adata, batch_key='batch',",
        "                                 Batch_list=[slice1, slice2],",
        "                                 iter_comb=[(0, 1)], device='cuda:0')",
        "staligner.train()",
        "adata_integrated = staligner.predicted()",
        "# Access integrated results",
        "integrated_embedding = adata_integrated.obsm['STAligner']"
    ],
    related=["space.Cal_Spatial_Net", "space.clusters", "space.svg"]
)
class pySTAligner(object):
    r"""STAligner for spatial transcriptomics data integration.
    
    STAligner is a deep learning method for integrating spatial transcriptomics
    data across different experimental conditions, technologies, and developmental
    stages. It combines graph neural networks with mutual nearest neighbors to
    preserve both transcriptional and spatial relationships during integration.

    The method works by:
    1. Constructing spatial neighborhood graphs
    2. Learning batch-invariant embeddings
    3. Aligning similar regions across batches
    4. Preserving spatial organization
    5. Enabling cross-condition comparison

    Parameters
    ----------
    adata : AnnData
        Combined multi-batch AnnData for integration.
    hidden_dims : list, default=[512, 30]
        Hidden dimensions of STAligner encoder.
    n_epochs : int, default=1000
        Total training epochs.
    lr : float, default=0.001
        Optimizer learning rate.
    batch_key : str, default='batch_name'
        Batch column in ``adata.obs``.
    key_added : str, default='STAligner'
        Output embedding key in ``adata.obsm``.
    gradient_clipping : float, default=5
        Max norm for gradient clipping.
    weight_decay : float, default=0.0001
        L2 regularization term.
    margin : float, default=1
        Margin used in triplet loss during alignment.
    verbose : bool, default=False
        Whether to print detailed training logs.
    random_seed : int, default=666
        Random seed for reproducibility.
    iter_comb : list, optional
        Batch-pair list for MNN comparison.
    knn_neigh : int, default=100
        K for mutual nearest-neighbor search.
    mnn_approx : bool or None, default=None
        Use hnswlib approximate-neighbor search. ``None`` selects it when
        available and otherwise falls back to exact scikit-learn neighbors.
    Batch_list : list, optional
        Per-batch AnnData list aligned to ``batch_key``.
    device : torch.device, default=auto cuda/cpu
        Device used for model training.

    Attributes:
        adata: AnnData
            Combined data containing all batches
        model: STAligner
            Neural network model for integration
        loader: DataLoader
            PyTorch geometric data loader
        device: torch.device
            Computing device (GPU/CPU)
        optimizer: torch.optim.Optimizer
            Adam optimizer for training

    Examples:
        >>> import scanpy as sc
        >>> import omicverse as ov
        >>> # Load data
        >>> adata1 = sc.read_visium(...)
        >>> adata2 = sc.read_visium(...)
        >>> # Construct spatial networks
        >>> ov.space.Cal_Spatial_Net(adata1, rad_cutoff=100)
        >>> ov.space.Cal_Spatial_Net(adata2, rad_cutoff=100)
        >>> # Combine data
        >>> adata = adata1.concatenate(adata2)
        >>> # Initialize STAligner
        >>> staligner = ov.space.pySTAligner(
        ...     adata=adata,
        ...     batch_key='batch',
        ...     Batch_list={'0': adata1, '1': adata2}
        ... )
        >>> # Train model
        >>> staligner.train()
        >>> # Get integrated embeddings
        >>> embeddings = staligner.predicted()
    """
    
    def __init__(self,adata,
                 hidden_dims: list = None,
                 n_epochs: int = 1000,
                 lr: float = 0.001,
                 batch_key: str = 'batch_name',
                 key_added: str = 'STAligner',
                 gradient_clipping: float = 5,
                 weight_decay: float = 0.0001,
                 margin: float = 1,
                 verbose: bool = False,
                 random_seed: int = 666,
                 iter_comb = None,
                 knn_neigh: int = 100,
                 mnn_approx = None,
                 Batch_list = None,
                 device = None,
                 batch_ids = None,
                 pretrain_epochs = None,
             ) -> None:
        r"""Initialize STAligner spatial integration model.
        
        This method sets up the STAligner model by:
        1. Processing input data
        2. Constructing graph neural networks
        3. Initializing optimization parameters
        4. Preparing batch alignment strategy

        Parameters
        ----------
        adata : AnnData
            Combined AnnData with batch labels in ``obs[batch_key]``.
        hidden_dims : list, default=[512, 30]
            Hidden dimensions of STAligner neural network.
        n_epochs : int, default=1000
            Number of training epochs.
        lr : float, default=0.001
            Learning rate for Adam.
        batch_key : str, default='batch_name'
            Batch label column in ``adata.obs``.
        key_added : str, default='STAligner'
            Output key for final embedding in ``adata.obsm``.
        gradient_clipping : float, default=5
            Maximum gradient norm.
        weight_decay : float, default=0.0001
            L2 regularization coefficient.
        margin : float, default=1
            Triplet-loss margin.
        verbose : bool, default=False
            Print detailed logs when ``True``.
        random_seed : int, default=666
            Random seed.
        iter_comb : list, optional
            Batch combinations for pairwise alignment.
        knn_neigh : int, default=100
            MNN neighbor count.
        mnn_approx : bool or None, default=None
            Whether to use hnswlib approximate neighbors. ``None`` uses it when
            importable and falls back to exact scikit-learn neighbors.
        Batch_list : list or mapping, optional
            Per-batch AnnData objects. A ``{batch_label: AnnData}`` mapping is
            safest when slices share the same Visium barcodes.
        device : torch.device, default=auto cuda/cpu
            Compute device for model.
        batch_ids : sequence, optional
            Explicit identity of each ``Batch_list`` item, in list order. This
            is required when two source slices have indistinguishable
            ``obs_names`` and cannot be inferred from a constant
            ``batch.obs[batch_key]`` column.
        pretrain_epochs : int, optional
            Number of STAGATE-only epochs before MNN alignment. Defaults to
            half of ``n_epochs`` (capped at 500), leaving at least half of short
            runs for actual alignment.

        Notes:
            - Requires pre-computed spatial networks
            - GPU acceleration recommended for large datasets
            - Batch_list order must match batch_key order
            - Memory usage scales with dataset size
            - Consider reducing knn_neigh for large datasets
        """
        hidden_dims = [512, 30] if hidden_dims is None else list(hidden_dims)
        if not isinstance(n_epochs, (int, np.integer)) or n_epochs < 2:
            raise ValueError(
                "`n_epochs` must be at least 2 so STAligner runs both its "
                "pretraining and MNN-alignment stages."
            )
        if batch_key not in adata.obs:
            raise KeyError(f"batch_key {batch_key!r} was not found in adata.obs.")
        if Batch_list is None:
            raise ValueError(
                "`Batch_list` is required and must contain one preprocessed AnnData "
                "per batch, each with `uns['adj']` from Cal_Spatial_Net()."
            )
        section_ids = np.array(adata.obs[batch_key].unique())
        if isinstance(Batch_list, Mapping):
            if batch_ids is not None:
                raise ValueError(
                    "Do not pass `batch_ids` when Batch_list is already a mapping."
                )
            missing_ids = [section_id for section_id in section_ids if section_id not in Batch_list]
            extra_ids = [key for key in Batch_list if key not in set(section_ids)]
            if missing_ids or extra_ids:
                raise ValueError(
                    "Batch_list mapping keys must exactly match adata.obs batch "
                    f"labels; missing={missing_ids}, extra={extra_ids}."
                )
            batch_ids = list(section_ids)
            Batch_list = [Batch_list[section_id] for section_id in section_ids]
        else:
            Batch_list = list(Batch_list)
        if len(Batch_list) < 2:
            raise ValueError("STAligner requires at least two batches in `Batch_list`.")
        too_small = [i for i, batch in enumerate(Batch_list) if batch.n_obs < 2]
        if too_small:
            raise ValueError(
                "STAligner triplet training requires at least two observations "
                f"per batch; too-small batch indices: {too_small}."
            )
        if pretrain_epochs is None:
            pretrain_epochs = min(500, max(1, n_epochs // 2))
        if (
            isinstance(pretrain_epochs, bool)
            or not isinstance(pretrain_epochs, (int, np.integer))
            or not 1 <= pretrain_epochs < n_epochs
        ):
            raise ValueError(
                "`pretrain_epochs` must be an integer satisfying "
                "1 <= pretrain_epochs < n_epochs."
            )

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        ) if device is None else torch.device(device)
        if len(section_ids) != len(Batch_list):
            raise ValueError(
                f"adata.obs[{batch_key!r}] contains {len(section_ids)} batches but "
                f"Batch_list contains {len(Batch_list)} objects."
            )
        if not adata.obs_names.is_unique:
            raise ValueError("`adata.obs_names` must be unique for MNN matching.")

        section_values = np.asarray(adata.obs[batch_key])
        combined_blocks = []
        row_start = 0
        for section_id in section_ids:
            positions = np.flatnonzero(section_values == section_id)
            expected = np.arange(row_start, row_start + len(positions))
            if not np.array_equal(positions, expected):
                raise ValueError(
                    "Rows in `adata` must be contiguous by batch in the "
                    "first-seen order of adata.obs[batch_key]."
                )
            combined_blocks.append(np.asarray(adata.obs_names[positions], dtype=str))
            row_start += len(positions)
        if row_start != adata.n_obs:
            raise ValueError("Could not account for every adata row by batch label.")

        resolved_batch_ids = None
        if batch_ids is not None:
            resolved_batch_ids = list(batch_ids)
            if len(resolved_batch_ids) != len(Batch_list):
                raise ValueError("`batch_ids` must have one entry per Batch_list item.")
        else:
            inferred = []
            for batch in Batch_list:
                if batch_key not in batch.obs or batch.obs[batch_key].isna().any():
                    inferred = []
                    break
                unique_ids = list(pd.unique(batch.obs[batch_key]))
                if len(unique_ids) != 1:
                    inferred = []
                    break
                inferred.append(unique_ids[0])
            if len(inferred) == len(Batch_list):
                resolved_batch_ids = inferred

        if resolved_batch_ids is not None and [
            str(value) for value in resolved_batch_ids
        ] != [str(value) for value in section_ids]:
            raise ValueError(
                "`batch_ids` (or per-batch constant batch labels) must match the "
                "first-seen order of adata.obs[batch_key]."
            )

        if resolved_batch_ids is None:
            for i, combined_names in enumerate(combined_blocks):
                compatible = [
                    j
                    for j, candidate in enumerate(Batch_list)
                    if _obs_names_match_with_optional_suffix(
                        candidate.obs_names,
                        combined_names,
                    )
                ]
                if compatible != [i]:
                    raise ValueError(
                        "Batch_list order cannot be verified because source "
                        "obs_names are ambiguous across slices. Pass explicit "
                        "`batch_ids` in Batch_list order, or add a constant "
                        f"{batch_key!r} column to each source AnnData."
                    )

        for i, (section_id, batch, combined_names) in enumerate(
            zip(section_ids, Batch_list, combined_blocks)
        ):
            if not batch.obs_names.is_unique:
                raise ValueError(f"Batch_list[{i}].obs_names must be unique.")
            if batch.n_obs != len(combined_names):
                raise ValueError(
                    f"Batch_list[{i}] has {batch.n_obs} rows but combined batch "
                    f"{section_id!r} has {len(combined_names)}."
                )
            source_names = np.asarray(batch.obs_names, dtype=str)
            if not _obs_names_match_with_optional_suffix(source_names, combined_names):
                raise ValueError(
                    f"Rows for batch {section_id!r} must match Batch_list[{i}].obs_names "
                    "in order. A uniform suffix added by AnnData.concatenate is allowed."
                )
        batch_adjs = [
            _validated_staligner_adjacency(batch, i)
            for i, batch in enumerate(Batch_list)
        ]
        if iter_comb is None:
            iter_comb = [(i, i + 1) for i in range(len(Batch_list) - 1)]
        else:
            iter_comb = [tuple(comb) for comb in iter_comb]
        if not iter_comb:
            raise ValueError("`iter_comb` must contain at least one batch pair.")
        for comb in iter_comb:
            if (
                len(comb) != 2
                or not all(isinstance(i, (int, np.integer)) for i in comb)
                or comb[0] >= comb[1]
                or min(comb) < 0
                or max(comb) >= len(Batch_list)
            ):
                raise ValueError(
                    "Each `iter_comb` entry must contain two valid Batch_list indices "
                    "in ascending order (i, j) with i < j."
                )
        if len(set(iter_comb)) != len(iter_comb):
            raise ValueError("`iter_comb` must not contain duplicate batch pairs.")
        reachable = {0}
        changed = True
        while changed:
            changed = False
            for left, right in iter_comb:
                if left in reachable and right not in reachable:
                    reachable.add(right)
                    changed = True
                elif right in reachable and left not in reachable:
                    reachable.add(left)
                    changed = True
        if reachable != set(range(len(Batch_list))):
            raise ValueError(
                "`iter_comb` must form one connected graph covering every batch; "
                f"uncovered batch indices: {sorted(set(range(len(Batch_list))) - reachable)}."
            )

        comm_gene = adata.var_names
        data_list = []
        for adata_tmp, adjacency in zip(Batch_list, batch_adjs):
            adata_tmp = adata_tmp[:, comm_gene].copy()   # line 268 avoid 'ArrayView'
            adata_tmp_X = adata_tmp.X.toarray() if hasattr(adata_tmp.X, 'toarray') else adata_tmp.X
            edge_index = np.nonzero(adjacency)
            data_list.append(
                Data(edge_index=torch.LongTensor(np.array([edge_index[0], edge_index[1]])),
                              prune_edge_index=torch.LongTensor(np.array([])),
                              x=torch.FloatTensor(adata_tmp_X)))

        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(random_seed))
        loader = DataLoader(
            data_list,
            batch_size=1,
            shuffle=True,
            generator=loader_generator,
        )

        self.loader=loader
        self.adata = adata
        self.data_list = data_list
        self.batch_adjs = batch_adjs
        self._loader_generator = loader_generator

        # hyper-parameters
        self.lr=lr
        self.section_ids = section_ids
        self.n_epochs = n_epochs
        self.pretrain_epochs = int(pretrain_epochs)
        self.weight_decay=weight_decay
        self.hidden_dims = hidden_dims
        self.key_added = key_added
        self.gradient_clipping = gradient_clipping
        self.random_seed = random_seed
        self.margin = margin
        self.verbose = verbose
        self.iter_comb = iter_comb
        self.knn_neigh = knn_neigh
        if mnn_approx is None:
            try:
                import hnswlib  # noqa: F401
            except (ImportError, OSError):
                self.mnn_approx = False
                warnings.warn(
                    "hnswlib is unavailable; STAligner will use exact "
                    "scikit-learn MNN search. Install hnswlib or pass "
                    "mnn_approx=False to silence this message.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                self.mnn_approx = True
        else:
            self.mnn_approx = bool(mnn_approx)
            if self.mnn_approx:
                try:
                    import hnswlib  # noqa: F401
                except (ImportError, OSError) as exc:
                    raise ImportError(
                        "mnn_approx=True requires hnswlib. Install it or use "
                        "mnn_approx=False for exact neighbor search."
                    ) from exc
        self.Batch_list = Batch_list
        self.batch_key = batch_key
        self._is_fitted = False
        STAligner, _ = _get_staligner_backend()
        fork_devices = []
        if self.device.type == 'cuda':
            fork_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(random_seed))
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(int(random_seed))
            self.model = STAligner(
                hidden_dims=[adata.X.shape[1], hidden_dims[0], hidden_dims[1]]
            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                          weight_decay=weight_decay)

        if verbose:
            print(self.model)

    def train(self):
        r"""Train the STAligner spatial integration model.
        
        This method performs two-stage training of the STAligner model:
        1. Pre-training with STAGATE to learn initial embeddings
        2. Fine-tuning with STAligner using triplet loss and MNN

        The training process:
        1. Sets random seeds for reproducibility
        2. Pre-trains with graph autoencoder
        3. Identifies mutual nearest neighbors
        4. Optimizes embeddings with triplet loss
        5. Monitors training progress
        6. Saves final embeddings

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes:
            - Progress shown if verbose=True
            - Uses GPU if available
            - Early stopping not implemented
            - Results stored in adata.obsm[key_added]
            - Memory usage increases during training
            - Consider batch size for large datasets
        """
        self._is_fitted = False
        rng = np.random.default_rng(self.random_seed)

        print('Pretrain with STAGATE...')
        pretrain_epochs = self.pretrain_epochs
        for epoch in tqdm(range(pretrain_epochs)):
            for batch in self.loader:
                self.model.train()
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                z, out = self.model(batch.x, batch.edge_index)

                loss = F.mse_loss(batch.x, out)  # +adv_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clipping,
                )
                self.optimizer.step()


        with torch.no_grad():
            z_list = []
            for batch in self.data_list:
                z, _ = self.model(
                    batch.x.to(self.device),
                    batch.edge_index.to(self.device),
                )
                z_list.append(z.cpu().detach().numpy())
        self.adata.obsm['STAGATE'] = np.concatenate(z_list, axis=0)


        _, create_dictionary_mnn = _get_staligner_backend()
        print('Train with STAligner...')
        for epoch in tqdm(range(pretrain_epochs, self.n_epochs)):
            if epoch % 100 == 0 or epoch == pretrain_epochs:
                if self.verbose:
                    print('Update spot triplets at epoch ' + str(epoch))
                with torch.no_grad():
                    z_list = []
                    for batch in self.data_list:
                        z, _ = self.model(
                            batch.x.to(self.device),
                            batch.edge_index.to(self.device),
                        )
                        z_list.append(z.cpu().detach().numpy())

                self.adata.obsm['STAGATE'] = np.concatenate(z_list, axis=0)

                pair_data_list = []

                for comb in self.iter_comb:
                    #print(comb)
                    i, j = comb[0], comb[1]
                    batch_pair = self.adata[self.adata.obs[self.batch_key].isin([self.section_ids[i],
                                                                                  self.section_ids[j]])]
                    effective_knn = min(
                        self.knn_neigh,
                        self.Batch_list[i].n_obs,
                        self.Batch_list[j].n_obs,
                    )
                    mnn_dict = create_dictionary_mnn(batch_pair, use_rep='STAGATE', batch_name=self.batch_key,
                                                           k=effective_knn,
                                                           iter_comb=None,
                                                           approx=self.mnn_approx,
                                                           verbose=0)

                    batchname_list = batch_pair.obs[self.batch_key]
                    cellname_by_batch_dict = dict()
                    for batch_id in range(len(self.section_ids)):
                        cellname_by_batch_dict[self.section_ids[batch_id]] = batch_pair.obs_names[
                            batch_pair.obs[self.batch_key] == self.section_ids[batch_id]].values
                    anchor_list = []
                    positive_list = []
                    negative_list = []
                    for batch_pair_name in mnn_dict.keys():  # pairwise compare for multiple batches
                        for anchor in mnn_dict[batch_pair_name].keys():
                            positive_spot = mnn_dict[batch_pair_name][anchor][0]
                            source_cells = cellname_by_batch_dict[batchname_list[anchor]]
                            negative_candidates = source_cells[source_cells != anchor]
                            if not len(negative_candidates):
                                continue
                            anchor_list.append(anchor)
                            positive_list.append(positive_spot)
                            negative_list.append(
                                negative_candidates[rng.integers(len(negative_candidates))]
                            )

                    if not anchor_list:
                        raise RuntimeError(
                            "STAligner found no usable mutual-nearest-neighbor "
                            f"anchors for batch pair {(i, j)}. Check preprocessing, "
                            "shared genes, and batch identity; the model was not "
                            "marked as fitted."
                        )

                    batch_as_dict = dict(zip(list(batch_pair.obs_names),
                                              range(0, batch_pair.shape[0])))
                    anchor_ind = list(map(lambda _: batch_as_dict[_], anchor_list))
                    positive_ind = list(map(lambda _: batch_as_dict[_], positive_list))
                    negative_ind = list(map(lambda _: batch_as_dict[_], negative_list))

                    edge_list_1 = np.nonzero(self.batch_adjs[i])

                    edge_list_2 = np.nonzero(self.batch_adjs[j])

                    batch_i_size = self.Batch_list[i].n_obs
                    edge_list_2 = (
                        edge_list_2[0] + batch_i_size,
                        edge_list_2[1] + batch_i_size,
                    )
                    edge_list = [edge_list_1, edge_list_2]

                    edge_pairs = [np.append(edge_list[0][0], edge_list[1][0]),
                                   np.append(edge_list[0][1], edge_list[1][1])]

                    pair_x = torch.cat(
                        [self.data_list[i].x, self.data_list[j].x],
                        dim=0,
                    ).clone()
                    pair_data_list.append(Data(
                        edge_index=torch.LongTensor(
                            np.array([edge_pairs[0], edge_pairs[1]])
                        ),
                        anchor_ind=torch.LongTensor(np.array(anchor_ind)),
                        positive_ind=torch.LongTensor(np.array(positive_ind)),
                        negative_ind=torch.LongTensor(np.array(negative_ind)),
                        x=pair_x,
                    ))

                # for temp in pair_data_list:
                #     temp.to(device)
                pair_loader = DataLoader(
                    pair_data_list,
                    batch_size=1,
                    shuffle=True,
                    generator=self._loader_generator,
                )

            for batch in pair_loader:
                self.model.train()
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                z, out = self.model(batch.x, batch.edge_index)
                mse_loss = F.mse_loss(batch.x, out)

                anchor_arr = z[batch.anchor_ind,]
                positive_arr = z[batch.positive_ind,]
                negative_arr = z[batch.negative_ind,]

                triplet_loss = torch.nn.TripletMarginLoss(margin=self.margin, p=2, reduction='sum')
                tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)

                loss = mse_loss + tri_output
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                self.optimizer.step()

        add_reference(self.adata,'STAligner','spatial integration with STAligner')
        self._is_fitted = True

    def predicted(self):
        r"""Generate and store the final embedding from trained STAligner model.

        Parameters
        ----------
        None

        Returns
        -------
        AnnData
            AnnData with integrated embedding in ``obsm[key_added]``.
        """ 
        if not self._is_fitted:
            raise RuntimeError("Call `train()` before requesting STAligner predictions.")
        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            z_list = []
            for batch in self.data_list:
                z, _ = self.model(
                    batch.x.to(self.device),
                    batch.edge_index.to(self.device),
                )
                z_list.append(z.cpu().detach().numpy())

        self.adata.obsm[self.key_added] = np.concatenate(z_list, axis=0)
        add_reference(self.adata,'STAligner','spatial integration with STAligner')
        return self.adata
# End-of-file (EOF)
