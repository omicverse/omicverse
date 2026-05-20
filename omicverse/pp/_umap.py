from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from datetime import datetime

import numpy as np
from sklearn.utils import check_array, check_random_state

from scanpy import settings
from ._compat import old_positionals, get_init_pos_from_paga
from ._neighbors import NeighborsView, _choose_representation
from .._settings import EMOJI, Colors, settings as ov_settings

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData

    from ._neighbors import _LegacyRandom

    _InitPos = Literal["paga", "spectral", "random"]


@old_positionals(
    "min_dist",
    "spread",
    "n_components",
    "maxiter",
    "alpha",
    "gamma",
    "negative_sample_rate",
    "init_pos",
    "random_state",
    "a",
    "b",
    "copy",
    "method",
    "neighbors_key",
)
def umap(  # noqa: PLR0913, PLR0915
    adata: AnnData,
    *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: _InitPos | np.ndarray | None = "spectral",
    random_state: _LegacyRandom = 0,
    a: float | None = None,
    b: float | None = None,
    method: Literal["umap", "rapids","torchdr","mde","pumap"] = "umap",
    key_added: str | None = None,
    neighbors_key: str = "neighbors",
    copy: bool = False,
) -> AnnData | None:
    r"""Embed the neighborhood graph using UMAP with multiple backend options.

    UMAP (Uniform Manifold Approximation and Projection) implementation with support
    for multiple computational backends including GPU-accelerated variants using
    TorchDR and MDE for improved performance on large datasets.

    Arguments:
        adata: Annotated data matrix of shape n_obs × n_vars
        min_dist (float): Minimum distance between embedded points (default: 0.5)
        spread (float): Scale at which embedded points are spread out (default: 1.0)
        n_components (int): Number of dimensions for the embedding (default: 2)
        maxiter (int): Number of optimization iterations (default: None)
        alpha (float): Initial learning rate for optimization (default: 1.0)
        gamma (float): Weighting for negative samples (default: 1.0)
        negative_sample_rate (int): Number of negative samples per positive sample (default: 5)
        init_pos: Initialization method for embedding positions (default: 'spectral')
        random_state (int): Random seed for reproducible results (default: 0)
        a (float): UMAP curve parameter (default: None)
        b (float): UMAP curve parameter (default: None)
        method (str): Implementation method: 'umap', 'torchdr', 'mde', 'rapids', 'pumap' (default: 'umap')
        key_added (str): Key for storing results in AnnData (default: None)
        neighbors_key (str): Key for accessing neighbor information (default: 'neighbors')
        copy (bool): Return copy instead of modifying in place (default: False)

    Returns:
        adata with UMAP embedding if copy=False, otherwise returns modified copy

    """
    adata = adata.copy() if copy else adata

    key_obsm, key_uns = ("X_umap", "umap") if key_added is None else [key_added] * 2

    if neighbors_key is None:  # backwards compat
        neighbors_key = "neighbors"
    if neighbors_key not in adata.uns:
        msg = f"Did not find .uns[{neighbors_key!r}]. Run `sc.pp.neighbors` first."
        raise ValueError(msg)

    print(f"\n{Colors.HEADER}{Colors.BOLD}{EMOJI['start']} UMAP Dimensionality Reduction:{Colors.ENDC}")
    print(f"   {Colors.CYAN}Mode: {Colors.BOLD}{ov_settings.mode}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Method: {Colors.BOLD}{method}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Components: {Colors.BOLD}{n_components}{Colors.ENDC}")
    print(f"   {Colors.CYAN}Min distance: {Colors.BOLD}{min_dist}{Colors.ENDC}")

    neighbors = NeighborsView(adata, neighbors_key)
    print(neighbors["params"])

    if "params" not in neighbors or neighbors["params"]["method"] != "umap":
        print(f"   {EMOJI['warning']} {Colors.WARNING}Connectivities matrix was not computed with UMAP method{Colors.ENDC}")

    with warnings.catch_warnings():
        # umap 0.5.0
        warnings.filterwarnings("ignore", message=r"Tensorflow not installed")
        import umap

    from umap.umap_ import find_ab_params, simplicial_set_embedding

    print(f"   {Colors.GREEN}{EMOJI['start']} Computing UMAP parameters...{Colors.ENDC}")
    
    # Convert random_state to proper RandomState object for UMAP and other scanpy functions
    random_state_processed = check_random_state(random_state)
    
    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)
    adata.uns[key_uns] = dict(params=dict(a=a, b=b))
    if isinstance(init_pos, str) and init_pos in adata.obsm:
        init_coords = adata.obsm[init_pos]
    elif isinstance(init_pos, str) and init_pos == "paga":
        init_coords = get_init_pos_from_paga(
            adata, random_state=random_state_processed, neighbors_key=neighbors_key
        )
    else:
        init_coords = init_pos  # Let umap handle it
    if hasattr(init_coords, "dtype"):
        init_coords = check_array(init_coords, dtype=np.float32, accept_sparse=False)

    if random_state != 0:
        adata.uns[key_uns]["params"]["random_state"] = random_state
    #print(neighbors.keys())
    neigh_params = neighbors["params"]
    X = _choose_representation(
        adata,
        use_rep=neigh_params.get("use_rep", None),
        n_pcs=neigh_params.get("n_pcs", None),
    )
    if method == "umap":
        print(f"   {Colors.GREEN}{EMOJI['start']} Computing UMAP embedding (classic method)...{Colors.ENDC}")
        # the data matrix X is really only used for determining the number of connected components
        # for the init condition in the UMAP embedding
        default_epochs = 10 if neighbors["connectivities"].shape[0] <= 10000 else 20
        n_epochs = default_epochs if maxiter is None else maxiter
        X_umap, _ = simplicial_set_embedding(
            data=X,
            graph=neighbors["connectivities"].tocoo(),
            n_components=n_components,
            initial_alpha=alpha,
            a=a,
            b=b,
            gamma=gamma,
            negative_sample_rate=negative_sample_rate,
            n_epochs=n_epochs,
            init=init_coords,
            random_state=random_state_processed,
            metric=neigh_params.get("metric", "euclidean"),
            metric_kwds=neigh_params.get("metric_kwds", {}),
            densmap=False,
            densmap_kwds={},
            output_dens=False,
            verbose=settings.verbosity > 3,
        )
    elif method == "torchdr":
        print(f"   {Colors.GREEN}{EMOJI['start']} Computing UMAP embedding (TorchDR GPU method)...{Colors.ENDC}")
        from torchdr import UMAP
        import torch
        from .._settings import get_optimal_device, prepare_data_for_device
        
        device = get_optimal_device(prefer_gpu=True, verbose=True)
        
        # Prepare data for MPS compatibility (float32 requirement)
        X = prepare_data_for_device(X, device, verbose=True)
        
        n_neighbors = neighbors["params"]["n_neighbors"]
        n_epochs = (
            500 if maxiter is None else maxiter
        )  # 0 is not a valid value for rapids, unlike original umap
        X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
        
        umap = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            max_iter=n_epochs,
            min_dist=min_dist,
            spread=spread,
            a=a,
            b=b,
            lr = alpha,
            device=device,
            random_state=random_state,
        )
        X_umap = umap.fit(X_contiguous).embedding_.detach().cpu().numpy()
        print(f"   {Colors.CYAN}💡 Using TorchDR UMAP on {device}{Colors.ENDC}")
        
        del umap
        import gc
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()

    elif method == "mde":
        print(f"   {Colors.GREEN}{EMOJI['start']} Computing UMAP embedding (MDE method)...{Colors.ENDC}")
        try:
            from pymde import MDE, constraints, penalties, preprocess
            import torch
        except ImportError as err:
            raise ImportError("Please install pymde package via `pip install pymde`") from err
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   {Colors.CYAN}Using device: {Colors.BOLD}{device}{Colors.ENDC}")
        
        # Convert random_state to torch seed
        import pymde
        pymde.seed(random_state)
        
        # Get neighbor graph from connectivities
        connectivities = neighbors["connectivities"]
        
        # Convert to PyTorch sparse tensor
        rows, cols = connectivities.nonzero()
        edges = torch.tensor(np.vstack([rows, cols]).T, dtype=torch.int64)
        
        # Filter out self-loops and ensure each edge appears only once
        mask = edges[:, 0] != edges[:, 1]
        edges = edges[mask]
        
        # Sort edges to ensure i < j
        sorted_edges = torch.zeros_like(edges)
        sorted_edges[:, 0] = torch.min(edges[:, 0], edges[:, 1])
        sorted_edges[:, 1] = torch.max(edges[:, 0], edges[:, 1])
        edges = torch.unique(sorted_edges, dim=0)
        
        # Create weights for edges based on connectivities
        weights = torch.ones(edges.shape[0], device=device)
        
        # Add dissimilar edges for better embedding
        n_similar = edges.shape[0]
        n_dissimilar = int(n_similar * min_dist)  # Use min_dist to scale number of dissimilar edges
        
        try:
            # Try to generate dissimilar edges
            dissimilar_edges = preprocess.dissimilar_edges(
                X.shape[0], 
                num_edges=int(n_dissimilar), 
                similar_edges=edges,
            )
            
            # Combine similar and dissimilar edges
            all_edges = torch.cat([edges, dissimilar_edges.to(edges.device)])
            
            # Create weights: positive for similar edges, negative for dissimilar
            all_weights = torch.cat([
                weights,  # Positive weights for similar edges
                -0.5 * torch.ones(dissimilar_edges.shape[0], device=device)  # Negative weights for dissimilar
            ])
            
            # Create PushAndPull distortion function
            distortion_function = penalties.PushAndPull(
                weights=all_weights,
                attractive_penalty=penalties.Log1p,
                repulsive_penalty=penalties.Log,
            )
            
            # Use the combined edges
            edges_to_use = all_edges
            
        except Exception as e:
            # Fallback to simple quadratic penalty if dissimilar edges fail
            print(f"   {EMOJI['warning']} {Colors.WARNING}Failed to generate dissimilar edges, using quadratic penalty{Colors.ENDC}")
            distortion_function = penalties.Quadratic(weights)
            edges_to_use = edges
        
        n_epochs = 500 if maxiter is None else maxiter
        
        # Create MDE problem with appropriate penalty
        mde = MDE(
            n_items=X.shape[0],
            embedding_dim=n_components,
            edges=edges_to_use.to(device),
            distortion_function=distortion_function,
            constraint=constraints.Standardized(),
            device=device
        )
        
        # Compute embedding with initial random state
        if isinstance(init_coords, np.ndarray):
            embedding = mde.embed(
                verbose=settings.verbosity > 3,
                max_iter=n_epochs,
                # Set random initial state if not using existing embedding
                X=None if init_coords is None else init_coords
            )
        else:
            embedding = mde.embed(
                verbose=settings.verbosity > 3,
                max_iter=n_epochs,
            )
        
        X_umap = embedding.cpu().numpy()
        
        # Clean up
        del mde, embedding, edges, weights
        if 'all_edges' in locals():
            del all_edges
        if 'all_weights' in locals():
            del all_weights
        if 'distortion_function' in locals():
            del distortion_function
    
        
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    elif method == "pumap":
        print(f"   {Colors.GREEN}{EMOJI['start']} Computing UMAP embedding (Parametric PyTorch method)...{Colors.ENDC}")
        try:
            from ..external.umap_pytorch.main import PUMAP
            import torch
        except ImportError as err:
            raise ImportError("PUMAP module not found in omicverse.external.umap_pytorch") from err

        # Determine device based on availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_gpus = 1 if torch.cuda.is_available() else 0
        print(f"   {Colors.CYAN}Using device: {Colors.BOLD}{device}{Colors.ENDC}")

        # Get neighbor parameters
        n_neighbors = neighbors["params"]["n_neighbors"]

        # Convert data to torch tensor
        X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X_contiguous)

        # The default MLP is tiny (50->200->200->200->2) so per-batch GPU
        # compute is dwarfed by Python loop + CUDA launch overhead until the
        # batch is in the tens of thousands. Sweep at 10k cells: 1024 -> 2.6s,
        # 4096 -> 0.6s, 16384 -> 0.39s, 65536 -> 0.32s. 16384 is the sweet
        # spot — fewer batches than 4096, but still enough optimizer steps
        # for convergence at any N.
        n_samples = X.shape[0]
        batch_size = 16384

        # `maxiter` is total edge-SGD graph passes (umap-learn semantics);
        # internally we split it into `training_epochs` chunks for early
        # stopping. Defaults match umap-learn (200 for N>10k, 500 below).
        total_graph_epochs = maxiter if maxiter is not None else (
            500 if n_samples <= 10_000 else 200
        )
        training_epochs = 4

        # Adam learning rate. UMAP `alpha` is the curve exponent, not a NN LR.
        learning_rate = 1e-3

        print(f"   {Colors.CYAN}Dataset: {Colors.BOLD}{n_samples} samples × {X.shape[1]} features{Colors.ENDC}")
        print(f"   {Colors.CYAN}Batch size: {Colors.BOLD}{batch_size}{Colors.ENDC}")
        print(f"   {Colors.CYAN}Learning rate: {Colors.BOLD}{learning_rate}{Colors.ENDC}")
        print(f"   {Colors.CYAN}Total graph passes: {Colors.BOLD}{total_graph_epochs}{Colors.ENDC} "
              f"{Colors.CYAN}(split into {Colors.BOLD}{training_epochs}{Colors.ENDC}{Colors.CYAN} training epochs){Colors.ENDC}")

        # Initialize PUMAP with early stopping
        pumap = PUMAP(
            encoder=None,  # Use default encoder
            decoder=None,  # No decoder needed for embedding only
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=neigh_params.get("metric", "euclidean"),
            n_components=n_components,
            beta=1.0,
            random_state=random_state,
            lr=learning_rate,
            epochs=training_epochs,
            graph_n_epochs=total_graph_epochs,
            batch_size=batch_size,
            num_workers=0,  # Avoid CUDA-multiprocessing issues
            num_gpus=num_gpus,
            match_nonparametric_umap=False,
            early_stopping=True,
            patience=2,  # In *training epochs* — each is a long edge pass.
            min_delta=1e-5,
        )

        # Reuse the GPU fuzzy graph if a prior ov.pp.neighbors(method='torch')
        # populated the cache — skips rebuilding KNN.
        from ._gpu_cache import cache_get
        precomputed = cache_get(
            adata, "connectivities", source_obj=neighbors["connectivities"],
        )
        print(f"   {Colors.CYAN}Training parametric UMAP model...{Colors.ENDC}")
        pumap.fit(X_tensor, precomputed_gpu_coo=precomputed)
        X_umap = pumap.transform(X_tensor)

        print(f"   {Colors.CYAN}💡 Using Parametric UMAP (PyTorch) on {device}{Colors.ENDC}")

        # Clean up
        del pumap, X_tensor
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    elif method == "rapids":
        msg = (
            "`method='rapids'` is deprecated. "
            "Use `rapids_singlecell.tl.louvain` instead."
        )
        warnings.warn(msg, FutureWarning, stacklevel=2)
        metric = neigh_params.get("metric", "euclidean")
        if metric != "euclidean":
            msg = (
                f"`sc.pp.neighbors` was called with `metric` {metric!r}, "
                "but umap `method` 'rapids' only supports the 'euclidean' metric."
            )
            raise ValueError(msg)
        from cuml import UMAP

        n_neighbors = neighbors["params"]["n_neighbors"]
        n_epochs = (
            100 if maxiter is None else maxiter
        )  # 0 is not a valid value for rapids, unlike original umap
        X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
        umap = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_epochs=n_epochs,
            learning_rate=alpha,
            init=init_pos,
            min_dist=min_dist,
            spread=spread,
            negative_sample_rate=negative_sample_rate,
            a=a,
            b=b,
            verbose=settings.verbosity > 3,
            random_state=random_state,
        )
        X_umap = umap.fit_transform(X_contiguous)
    adata.obsm[key_obsm] = X_umap  # annotate samples with UMAP coordinates
    print(f"\n{Colors.GREEN}{EMOJI['done']} UMAP Dimensionality Reduction Completed Successfully!{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Embedding shape: {Colors.BOLD}{X_umap.shape[0]:,}{Colors.ENDC}{Colors.GREEN} cells × {Colors.BOLD}{X_umap.shape[1]}{Colors.ENDC}{Colors.GREEN} dimensions{Colors.ENDC}")
    print(f"   {Colors.GREEN}✓ Results added to AnnData object:{Colors.ENDC}")
    print(f"     {Colors.CYAN}• '{key_obsm}': {Colors.BOLD}UMAP coordinates{Colors.ENDC}{Colors.CYAN} (adata.obsm){Colors.ENDC}")
    print(f"     {Colors.CYAN}• '{key_uns}': {Colors.BOLD}UMAP parameters{Colors.ENDC}{Colors.CYAN} (adata.uns){Colors.ENDC}")
    return adata if copy else None



