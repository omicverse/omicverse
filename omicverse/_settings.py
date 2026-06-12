from ._registry import register_function
from ._optional import normalize_torch_device

class omicverseConfig:

    def __init__(self,mode='cpu'):
        self.mode = mode
        self.seed = None
        from .utils._analytics_sender import send_analytics_full_silent
        import datetime
        test_id_full = f"FULL-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        send_analytics_full_silent(test_id_full)

    def set_seed(self, seed: int = 0, *, deterministic: bool = False,
                 verbose: bool = True) -> int:
        """Set the global random seed for reproducible results.

        Seeds Python / NumPy / PyTorch (CPU+CUDA) / MLX and stores the value
        in ``self.seed``. See :func:`omicverse.set_seed` for details.
        """
        from .utils._seed import set_seed as _set_seed
        return _set_seed(seed, deterministic=deterministic, verbose=verbose)

    @register_function(
        aliases=["GPU初始化", "gpu_init", "gpu_mode", "GPU模式", "rapids_init"],
        category="utils",
        description="Initialize GPU mode with RAPIDS for accelerated single-cell analysis",
        examples=[
            "# Initialize GPU mode with default settings",
            "ov.settings.gpu_init()",
            "# Custom GPU initialization",
            "ov.settings.gpu_init(managed_memory=False, pool_allocator=True)",
            "# Use specific GPU device", 
            "ov.settings.gpu_init(devices=1)",
            "# Check current mode",
            "print(f'Current mode: {ov.settings.mode}')"
        ],
        related=["settings.cpu_init", "settings.cpu_gpu_mixed_init", "pp.anndata_to_GPU"]
    )
    def gpu_init(self,managed_memory=True,pool_allocator=True,devices=0):
        r"""Initialize GPU mode with RAPIDS for accelerated single-cell analysis.

        Parameters
        ----------
        managed_memory:bool, optional
            Enable NVIDIA Unified Memory to support oversubscription for large datasets.
        pool_allocator:bool, optional
            Enable RMM memory pool allocator for faster repeated GPU allocations.
        devices:int|list[int], optional
            CUDA device index (or indices) to register in RMM.

        Returns
        -------
        None
            Sets ``self.mode`` to ``'gpu'`` and configures RAPIDS/CuPy allocators.

        Examples:
            >>> import omicverse as ov
            >>> # Initialize GPU mode with default settings
            >>> ov.settings.gpu_init()
            >>> # Custom GPU initialization
            >>> ov.settings.gpu_init(managed_memory=False, pool_allocator=True)
        """

        import scanpy as sc
        import cupy as cp

        import time
        import rapids_singlecell as rsc

        import warnings
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        rmm.reinitialize(
            managed_memory=managed_memory,  # Allows oversubscription
            pool_allocator=pool_allocator,  # default is False
            devices=devices,  # GPU device IDs to register. By default registers only GPU 0.
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)
        print('GPU mode activated')
        self.mode = 'gpu'
    
    def cpu_init(self):
        print('CPU mode activated')
        self.mode = 'cpu'
    
    @register_function(
        aliases=["CPU-GPU混合模式", "cpu_gpu_mixed_init", "mixed_mode", "GPU混合模式", "mixed_init"],
        category="utils",
        description="Initialize CPU-GPU mixed mode for accelerated single-cell analysis",
        examples=[
            "# Initialize mixed mode for better performance", 
            "ov.settings.cpu_gpu_mixed_init()",
            "# Use mixed mode with preprocessing",
            "ov.settings.cpu_gpu_mixed_init()",
            "adata = ov.pp.qc(adata)  # Automatically uses mixed mode",
            "# Check current mode",
            "print(f'Current mode: {ov.settings.mode}')"
        ],
        related=["settings.gpu_init", "settings.cpu_init", "pp.qc", "pp.preprocess"]
    )
    def cpu_gpu_mixed_init(self):
        r"""Initialize CPU-GPU mixed mode for accelerated single-cell analysis.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Sets ``self.mode`` to ``'cpu-gpu-mixed'`` and reports detected accelerators.

        Examples:
            >>> import omicverse as ov
            >>> # Initialize mixed mode for better performance
            >>> ov.settings.cpu_gpu_mixed_init()
            >>> # Use mixed mode with preprocessing
            >>> ov.settings.cpu_gpu_mixed_init()
            >>> adata = ov.pp.qc(adata)  # Automatically uses mixed mode
        """
        print('CPU-GPU mixed mode activated')
        
        # Detect available GPU accelerators for mixed mode
        if torch is not None:
            available_devices = []
            if torch.cuda.is_available():
                available_devices.append("CUDA")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                available_devices.append("MPS")
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                available_devices.append("ROCm")
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                available_devices.append("XPU")
            
            if available_devices:
                print(f'Available GPU accelerators: {", ".join(available_devices)}')
            else:
                print('No GPU accelerators detected - fallback to CPU')
        
        self.mode = 'cpu-gpu-mixed'
    




import subprocess
try:
    import torch  # Optional GPU dependency
except ImportError:  # pragma: no cover - optional dependency
    torch = None

def check_acceleration_packages():
    """
    Check which acceleration packages are installed and provide installation guidance.

    Returns
    -------
    dict
        Dictionary containing installation status and recommendations
    """
    status = {
        'mlx': {'installed': False, 'recommended': False, 'device': 'mps'},
        'torch': {'installed': False, 'recommended': False, 'device': 'any'}
    }

    # Check PyTorch
    try:
        import torch
        status['torch']['installed'] = True
        status['torch']['recommended'] = True
    except ImportError:
        pass

    # Check MLX
    try:
        import mlx.core as mx
        status['mlx']['installed'] = True
        if mx.metal.is_available():
            status['mlx']['recommended'] = True
    except ImportError:
        if torch is not None and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            status['mlx']['recommended'] = True

    return status


def print_acceleration_status(verbose=True):
    """
    Print the current acceleration package status and installation recommendations.
    
    Parameters
    ----------
    verbose:bool
        Whether to print detailed information
    """
    status = check_acceleration_packages()
    
    print(f"{Colors.BLUE}🚀 Omicverse Acceleration Status:{Colors.ENDC}")
    print("=" * 50)
    
    # PyTorch status
    if status['torch']['installed']:
        print(f"{Colors.GREEN}✅ PyTorch: Installed{Colors.ENDC}")
        if verbose:
            try:
                import torch
                print(f"   Version: {torch.__version__}")
                if torch.cuda.is_available():
                    print(f"   CUDA: Available ({torch.cuda.device_count()} device(s))")
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    print(f"   MPS: Available (Apple Silicon)")
                print(f"   {Colors.CYAN}Note: Using built-in torch_pca for GPU-accelerated PCA{Colors.ENDC}")
            except:
                pass
    else:
        print(f"{Colors.WARNING}⚠️ PyTorch: Not installed{Colors.ENDC}")
        print(f"   {Colors.CYAN}Install with: pip install torch{Colors.ENDC}")

    # MLX status
    if status['mlx']['installed']:
        print(f"{Colors.GREEN}✅ MLX: Installed{Colors.ENDC}")
        if verbose:
            try:
                import mlx.core as mx
                if mx.metal.is_available():
                    print(f"   Metal: Available (Apple Silicon GPU)")
                else:
                    print(f"   Metal: Not available")
            except:
                pass
    elif status['mlx']['recommended']:
        print(f"{Colors.WARNING}⚠️ MLX: Not installed (Recommended for Apple Silicon){Colors.ENDC}")
        print(f"   {Colors.CYAN}Install with: pip install mlx{Colors.ENDC}")
        print(f"   {Colors.CYAN}For Apple Silicon GPU acceleration{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}❌ MLX: Not installed{Colors.ENDC}")
    
    print("=" * 50)
    
    # Recommendations
    recommendations = []
    if not status['torch']['installed']:
        recommendations.append("Install PyTorch for GPU support (includes built-in torch_pca)")
    if status['mlx']['recommended'] and not status['mlx']['installed']:
        recommendations.append("Install MLX for Apple Silicon GPU acceleration")

    if recommendations:
        print(f"{Colors.CYAN}💡 Recommendations:{Colors.ENDC}")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print(f"{Colors.GREEN}🎉 All recommended packages are installed!{Colors.ENDC}")


def get_optimal_device(prefer_gpu=True, verbose=False):
    """
    Get the optimal PyTorch device based on available hardware.
    Now includes acceleration package status checking.

    Priority order:
    1. CUDA (NVIDIA GPUs) - uses built-in torch_pca for GPU acceleration
    2. MPS (Apple Silicon) - MLX recommended for optimal performance
    3. ROCm (AMD GPUs)
    4. XPU (Intel GPUs)
    5. CPU (fallback)

    Parameters
    ----------
    prefer_gpu:bool
        Whether to prefer GPU over CPU when available
    verbose:bool
        Whether to print device selection information and acceleration status

    Returns
    -------
    torch.device
        The optimal device for computation
    """
    if torch is None:
        if verbose:
            print("PyTorch not available, using CPU")
            print(f"{Colors.WARNING}💡 Install PyTorch for GPU support: pip install torch{Colors.ENDC}")
        return "cpu"
    
    if not prefer_gpu:
        if verbose:
            print("GPU preference disabled, using CPU")
        return torch.device("cpu")
    
    # Check acceleration packages if verbose
    if verbose:
        #print_acceleration_status(verbose=True)
        print()  # Add spacing
    
    # Check devices in priority order
    if torch.cuda.is_available():
        device = normalize_torch_device("cuda")
        if verbose:
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
            print(f"{Colors.GREEN}✅ Using built-in torch_pca for GPU-accelerated PCA{Colors.ENDC}")
        return device
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Using Apple Silicon MPS device (note: float32 required)")
            # Check if MLX is available for optimal performance
            try:
                import mlx.core as mx
                if mx.metal.is_available():
                    print(f"{Colors.GREEN}✅ MLX available for Apple Silicon GPU acceleration{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}⚠️ MLX Metal not available{Colors.ENDC}")
            except ImportError:
                print(f"{Colors.WARNING}⚠️ MLX not installed - install with: pip install mlx{Colors.ENDC}")
        return device
    
    if hasattr(torch.version, 'hip') and torch.version.hip is not None and torch.cuda.is_available():
        # ROCm available
        device = normalize_torch_device("cuda")  # ROCm uses cuda interface
        if verbose:
            print("Using AMD ROCm device")
        return device
    
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        if verbose:
            print("Using Intel XPU device")
        return device
    
    # Fallback to CPU
    device = torch.device("cpu")
    if verbose:
        print("No GPU available, using CPU")
        print(f"{Colors.CYAN}💡 For GPU acceleration, ensure PyTorch is installed with GPU support{Colors.ENDC}")
    return device

def prepare_data_for_device(X, device, verbose=False):
    """
    Prepare data for a specific device, handling device-specific requirements.
    
    Parameters
    ----------
    X:array-like
        Input data (numpy array, sparse matrix, or other array-like)
    device:torch.device
        Target device
    verbose:bool
        Whether to print conversion information
        
    Returns
    -------
    X_prepared:array-like
        Data prepared for the target device
    """
    import numpy as np
    from scipy import sparse
    
    # Handle MPS float64 limitation
    if hasattr(device, 'type') and device.type == 'mps':
        # Handle numpy arrays
        if hasattr(X, 'dtype') and X.dtype == np.float64:
            if verbose:
                print("   Converting float64 to float32 for MPS compatibility")
            X = X.astype(np.float32)
        
        # Handle sparse matrices
        elif sparse.issparse(X) and X.dtype == np.float64:
            if verbose:
                print("   Converting sparse matrix float64 to float32 for MPS compatibility")
            X = X.astype(np.float32)
        
        # Handle AnnData objects
        elif hasattr(X, 'X'):
            if hasattr(X.X, 'dtype') and X.X.dtype == np.float64:
                if verbose:
                    print("   Converting AnnData.X float64 to float32 for MPS compatibility")
                if sparse.issparse(X.X):
                    X.X = X.X.astype(np.float32)
                else:
                    X.X = X.X.astype(np.float32)
        
        # Handle other array-like objects with dtype
        elif hasattr(X, 'astype') and hasattr(X, 'dtype') and X.dtype == np.float64:
            if verbose:
                print("   Converting array float64 to float32 for MPS compatibility")
            X = X.astype(np.float32)
    
    return X

def check_reference_key(adata):
    if 'REFERENCE_MANU' not in adata.uns.keys():
        adata.uns['REFERENCE_MANU']={}

def add_reference(adata,reference_name,reference_content):
    check_reference_key(adata)
    adata.uns['REFERENCE_MANU']['omicverse']='This analysis is performed with omicverse framework.'
    adata.uns['REFERENCE_MANU'][reference_name]=reference_content

reference_dict = {
    'omicverse':'Zeng, Z., Ma, Y., Hu, L., Tan, B., Liu, P., Wang, Y., ... & Du, H. (2024). OmicVerse: a framework for bridging and deepening insights across bulk and single-cell sequencing. Nature Communications, 15(1), 5983.',
    'scvi': 'Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. Nature methods, 15(12), 1053-1058.',
    'scanpy': 'Wolf, F. A., Angerer, P., & Theis, F. J. (2018). SCANPY: large-scale single-cell gene expression data analysis. Genome biology, 19, 1-5.',
    'dynamicTreeCut': 'Langfelder, P., Zhang, B., & Horvath, S. (2008). dynamicTreeCut: An R package for detecting clusters in a dynamic tree cut. Bioinformatics, 24(5), 719-720.',
    'scDrug': 'Guo, C., Wang, J., Liu, L., Xu, L., Chen, Y., Yu, G., ... & Zhang, G. (2022). scDrug: A single-cell multi-omics knowledgebase for drug discovery. iScience, 25(12), 105550.',
    'MOFA': 'Argelaguet, R., Velten, B., Arnol, D., Miller, M., Marioni, J. C., & Stegle, O. (2020). MOFA: A new tool for dissecting omics data. Genome biology, 21(1), 1-17.',
    'COSG': 'Jian, J., Hu, T., Jin, Z., Zhang, Z., Xie, J., & Zhang, Y. (2022). COSG: universal marker gene detection for unsupervised single-cell RNA sequencing data analysis. Briefings in Bioinformatics, 23(1), bbab579.',
    'CellphoneDB': 'Efremova, M., Vento-Tormo, M., Teichmann, S. A., & Vento-Tormo, R. (2020). CellPhoneDB: inferring cell–cell communication from combined single-cell and spatial transcriptomics data. Nature Protocols, 15(4), 1484-1506.',
    'AUCell': 'Aibar, S., González-Blas, C. B., Moerman, V., Huynh-Thu, V. A., Tritschler, F., van den Berge, K., ... & Aerts, S. (2017). AUCell: an R package to quantify the activity of gene sets in single-cell RNA-seq data. Bioinformatics, 34(18), 3006-3008.',
    'Bulk2Space': 'Fan, K., Li, P., Guo, M., Ding, X., Wang, Y., Zhang, W., & Zhang, W. (2022). Bulk2Space: Decoding spatial gene expression from bulk RNA-seq data. Nature Communications, 13(1), 6667.',
    'SCSA': 'Cao, H., Ma, D., Wang, Y., Zhu, C., Wang, H., Lu, X., ... & Chen, Y. (2020). SCSA: Single-cell RNA-Seq analysis for cell-type annotation and spatial cell-cell interaction prediction. Frontiers in Genetics, 11, 490.',
    'WGCNA': 'Langfelder, P., & Horvath, S. (2008). WGCNA: an R package for weighted correlation network analysis. BMC bioinformatics, 9(1), 559.',
    'StaVIA': 'Stassen, B., Vriens, A., De Coninck, T., Thienpont, B., Wouters, J., & Saelens, W. (2021). StaVIA: Spatiotemporal Variational Inference and Alignment. Nature Communications, 12(1), 5670.',
    'pyDEseq2': 'Lemoine, J., & Nicolas, J. C. (2022). pyDESeq2: A Python package for differential expression analysis of RNA sequencing data. bioRxiv, 2022-12.',
    'NOCD': 'Shchur, O., & Hein, M. (2019). Community detection in graphs with heterophilic and overlapping communities. arXiv preprint arXiv:1909.12201.',
    'SIMBA': 'Fan, X., Hu, M., He, S., Cheng, H., Wang, J., Yang, F., ... & Pinello, L. (2023). SIMBA: a tool for single-cell multi-modal data analysis. Nature Methods, 20(8), 1261-1271.',
    'GLUE': 'Cao, Z. J., Gao, M., & Zhang, K. (2022). GLUE: a graph-linked unified embedding for single-cell multi-omics integration. Nature Methods, 19(5), 589-599.',
    'MetaTiME': 'Zhang, Y., Li, M., & Zhang, J. (2023). MetaTiME: An interpretable meta-learning framework for estimating tumor immune microenvironment from bulk transcriptomics. Nature Communications, 14(1), 2636.',
    'TOSICA': 'Han, Y., Han, H., Li, G., Zang, Z., Zang, Z., Chen, G., ... & Chen, T. (2023). TOSICA: a robust tool for identifying and characterizing tumor-specific immunogenic cell-cell interactions. Nature Communications, 14(1), 585.',
    'Harmony': 'Korsunsky, I., Millard, K., Fan, J., Slowikowski, K., Zhang, F., Wei, K., ... & Park, P. J. (2019). Harmony: efficient integration of single-cell data from multiple experiments. Nature Methods, 16(12), 1217-1222.',
    'Scanorama': 'Hie, B., Bryant, E., & Yuan, G. C. (2019). Scanorama: a fast and accurate integration algorithm for single-cell RNA-seq data. Nature Methods, 16(7), 676-682.',
    'Combat': 'Johnson, W. E., Li, C., & Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics, 8(1), 118-127.',
    'TAPE': 'Chen, L., Wu, T., Liu, X., Liu, C., Yang, W., & Zhang, Y. (2022). TAPE: a Transformer-based deep learning method for predicting single-cell trajectories. Nature Communications, 13(1), 7119.',
    'SEACells': 'Brouwer, S., Dvir, R., & Peer, D. (2023). SEACells: a computational method for discovering and characterizing cell states from single-cell data. Nature Biotechnology, 41(6), 756-764.',
    'Palantir': 'Setty, M., Tadmor, M. D., Reich-Zeliger, S., Angel, O., Salame, T. M., McMahon, A. P., & Peer, D. (2019). Palantir: A diffusion-pseudotime-based trajectory inference method for single-cell RNA sequencing data. Nature Biotechnology, 37(8), 851-860.',
    'STAGATE': 'Jiang, Y., Li, S., Wang, Y., Zhang, C., Cheng, S., Lai, Z., ... & Qiu, Q. (2022). STAGATE: Spatially resolved transcriptomics analysis using graph attention autoencoder. Nature Communications, 13(1), 1774.',
    'MIRA': 'Liu, Q., Zhang, L., Wang, J., Wang, Z., Xie, J., Wang, S., ... & Han, X. (2022). MIRA: a method for integrating single-cell and spatial transcriptomics data. Nature Methods, 19(10), 1276-1286.',
    'Tangram': 'Biancalani, T., Scalia, G., Buffoni, L., Avasthi, R., Tiwary, S., Sanger, S., ... & Regev, A. (2021). Tangram: spatial mapping of single-cell transcriptomes by integrating spatial transcriptomics and single-cell RNA sequencing data. Nature Methods, 18(12), 1435-1440.',
    'STAligner': 'Xu, R., Yang, J., Zhao, L., Lu, R., Zhang, S., Liu, C., & Zhou, X. (2023). STAligner: an unsupervised deep learning model for integrating spatial transcriptomics data. Cell & Bioscience, 13(1), 173.',
    'CEFCON': 'Pan, W., & Zhang, W. (2023). CEFCON: a computational method for identifying functional conservation of enhancers across species. Nature Communications, 14(1), 7954.',
    'PyComplexHeatmap': 'Ding, W. B. (2022). PyComplexHeatmap: A Python package for drawing Complex Heatmaps. International Journal of Molecular Sciences, 23(23), 15004.',
    'STT': 'Zhou, Y., Gao, Y., Lv, Y., Zeng, T., & Liu, C. (2024). STT: Integrating spatial transcriptomics with single-cell transcriptomics for cell type mapping. Nature Methods, 21(5), 896-905.',
    'SLAT': 'Zhao, C., & Zhang, W. (2023). SLAT: a spatial and lineage-aware trajectory inference framework for single-cell spatial transcriptomics. Nature Communications, 14(1), 7247.',
    'GPTCelltype': 'Su, W., Yan, Y., Liu, W., Zhao, J., Chen, Z., Li, S., ... & Han, S. (2024). GPTCelltype: accurate and interpretable cell type annotation based on gene ontology and large language models. Nature Methods, 21(5), 882-895.',
    'PROST': 'Wang, Y., Li, S., Song, M., Wang, Y., Liang, J., Zhang, L., ... & Tang, F. (2024). PROST: prediction of spatial transcriptomics based on single-cell RNA sequencing data. Nature Communications, 15(1), 696.',
    'CytoTrace2': 'Shaham, O., & Buenrostro, J. D. (2024). CytoTrace2: predicting cell fate trajectories using single-cell RNA sequencing data. bioRxiv, 2024-03.',
    'GraphST': 'Ma, X., Du, S., Sun, S., Wang, H., & Chen, J. (2023). GraphST: a Graph Neural Network for Spatial Transcriptomics analysis. Nature Communications, 14(1), 1269.',
    'COMPOSITE': 'Xu, Z., Xu, H., Zhou, Z., Lin, Y., & Chen, X. (2024). COMPOSITE: a computational framework for the prediction of cell-type-specific chromatin accessibility from bulk data. Nature Communications, 15(1), 4991.',
    'mellon': 'Pang, A., Galdos, R., Zhang, J., & Setty, M. (2024). mellon: a computational method for integrating spatial and single-cell transcriptomics data to characterize cellular neighborhoods. Nature Methods, 21(8), 1437-1446.',
    'starfysh': 'Azizi, A., Hie, B., & Yuan, G. C. (2024). starfysh: a spatial transcriptomics data analysis toolbox for cell type deconvolution and enhanced visualization. Nature Biotechnology, 42(5), 780-789.',
    'COMMOT': 'Cang, Z., & Nie, Q. (2022). COMMOT: a computational framework for inferring cell-cell communication from single-cell RNA sequencing data. Nature Methods, 19(12), 1629-1640.',
    'flowsig': 'Almet, A. A., & Chen, Y. (2024). flowsig: an R package for single-cell gene signature scoring based on flow cytometry data. Nature Methods, 21(9), 1667-1670.',
    'pyWGCNA': 'Mirzarahimi, M., & Mortazavi, A. (2023). PyWGCNA: an advanced Python package for weighted gene co-expression network analysis. Bioinformatics, 39(7), btad415.',
    'CAST': 'Lu, L., Chen, Y., Yu, D., Jiang, Y., Peng, M., Li, Y., ... & Wang, X. (2024). CAST: A computational framework for identifying cell-type-specific transcription factor regulons from single-cell chromatin accessibility data. Nature Methods, 21(11), 2055-2066.',
    'scMulan': 'Bian, C., & Lu, Q. (2023). scMulan: an R package for multi-omics integration and visualization of single-cell data. In Single-Cell Omics (pp. 953-976). New York, NY: Springer US.',
    'cellANOVA': 'Zhang, J., Han, X., Li, X., Liu, C., Cao, Z., & Chen, L. (2024). cellANOVA: a statistical framework for differential expression analysis in single-cell RNA sequencing data. Nature Biotechnology, 42(12), 2038-2049.',
    'BINARY': 'Lin, S., Zhu, Y., Li, Y., Wang, P., Yang, Y., & Chen, G. (2024). BINARY: a tool for integrative analysis of single-cell RNA-seq and spatial transcriptomics data. Cell Reports Methods, 4(7), 100778.',
    'GASTON': 'Ding, J., Fu, Y., Wang, B., & Raphael, B. J. (2024). GASTON: Graph-based analysis of spatial transcriptomics data with non-parametric cell type decomposition. Nature Methods, 21(11), 2039-2048.',
    'pertpy': 'Deinzer, C., Lotz, M., Stoeckius, M., Bojar, P., & Theis, F. J. (2024). pertpy: A Python framework for comprehensive perturbation analysis in single-cell omics. bioRxiv, 2024-08.',
    'inmoose': '', # No readily available publication found with the provided link.
    'memento': 'Liu, S., & Ye, C. J. (2024). memento: A Python package for inference of interpretable interactions from single-cell transcriptomics data. Cell, 187(20), 4983-5000.e22.',
    'Wilcoxon':'Cuzick, J. (1985). A Wilcoxon‐type test for trend. Statistics in medicine, 4(1), 87-90.',
    'T-test':'Kim, T. K. (2015). T test as a parametric statistic. Korean journal of anesthesiology, 68(6), 540-546.',
    'GSEApy':'Fang, Z., Liu, X., & Peltz, G. (2023). GSEApy: a comprehensive package for performing gene set enrichment analysis in Python. Bioinformatics, 39(1), btac757.',
    'leiden':'Traag, V. A., Waltman, L., & Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific reports, 9(1), 1-12.',
    'louvain':'Blondel, V. D., Guillaume, J.-L., Lambiotte, R. & Lefebvre, E. Fast unfolding of communities in large networks. J. Stat. Mech. Theory Exp. 10008, 6, https://doi.org/10.1088/1742-5468/2008/10/P10008 (2008).',
    'GMM':'Bond, S. R., Hoeffler, A., & Temple, J. R. (2001). GMM estimation of empirical growth models. Available at SSRN 290522.',
    'mclust':'Fraley, C., & Raftery, A. E. (1998). MCLUST: Software for model-based cluster and discriminant analysis. Department of Statistics, University of Washington: Technical Report, 342, 1312.',
    'schist':'Morelli, L., Giansanti, V., & Cittaro, D. (2021). Nested Stochastic Block Models applied to the analysis of single cell data. BMC bioinformatics, 22, 1-19.',
    'umap':'McInnes, L., Healy, J., & Melville, J. (2018). Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.',
    'pymde':'Agrawal, A., Ali, A., & Boyd, S. (2021). Minimum-distortion embedding. Foundations and Trends® in Machine Learning, 14(3), 211-378.',
    'tsne':'Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(11).',
    'sctour':'Qian Li, scTour: a deep learning architecture for robust inference and accurate prediction of cellular dynamics, 2023, Genome Biology',
    'Concord':'Zhu, Q., Jiang, Z., Zuckerman, B. et al. Revealing a coherent cell-state landscape across single-cell datasets with CONCORD. Nat Biotechnol (2026).',
    'Banksy':'Singhal, V., Chou, N., Lee, J. et al. BANKSY unifies cell typing and tissue domain segmentation for scalable spatial omics data analysis. Nat Genet 56, 431–441 (2024). https://doi.org/10.1038/s41588-024-01664-3'
}

@register_function(
    aliases=['生成参考表', 'generate_reference_table', 'reference table', 'annotation reference table'],
    category="core",
    description="Generate a standardized reference table from AnnData feature metadata for downstream annotation, mapping, and report generation workflows.",
    prerequisites={'optional_functions': ['utils.get_gene_annotation']},
    requires={'var': ['gene symbols or feature annotation']},
    produces={'uns': ['reference_table']},
    auto_fix='none',
    examples=['ov.generate_reference_table(adata)'],
    related=['utils.get_gene_annotation', 'datasets.download_data', 'single.generate_scRNA_report']
)
def generate_reference_table(adata):
    """
    Generate a standardized reference table from ``adata.uns['REFERENCE_MANU']``.

    Parameters
    ----------
    adata:AnnData
        AnnData object that stores method references in ``uns['REFERENCE_MANU']``.

    Returns
    -------
    pandas.DataFrame|None
        A table with columns ``method``, ``content``, and ``reference``.
        Returns ``None`` when no reference metadata is available.
    """
    import pandas as pd
    if 'REFERENCE_MANU' not in adata.uns.keys():
        return None
    reference_table=pd.DataFrame(columns=['method','reference'])
    reference_dic=adata.uns['REFERENCE_MANU']
    for ref in reference_dic:
        if ref in reference_dict.keys():
            reference_table=pd.concat([reference_table, pd.DataFrame({'method':[ref],
                                                                      'content':[reference_dic[ref]],
                                                                      'reference':[reference_dict[ref]]})], ignore_index=True)
        else:
            reference_table=pd.concat([reference_table, pd.DataFrame({'method':[ref],
                                                                      'content':[reference_dic[ref]],
                                                                      'reference':['']})], ignore_index=True)
    return reference_table


def print_gpu_usage_color(bar_length: int = 30):
    """
    Print a colorized memory‐usage bar for each GPU (NVIDIA, AMD, Intel, Apple Silicon).

    Parameters
    ----------
    bar_length:int
        Total characters in each usage bar (filled + empty).
    """
    # ANSI escape codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GREY = '\033[90m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    if torch is None:
        print(f"{RED}PyTorch not available for GPU monitoring.{RESET}")
        return

    # Try different GPU backends
    gpu_found = False
    
    # 1. NVIDIA CUDA GPUs
    if torch.cuda.is_available():
        print(f"{BLUE}NVIDIA CUDA GPUs detected:{RESET}")
        gpu_found = True
        
        # Try nvidia-smi first (most detailed info)
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,name",
                "--format=csv,noheader,nounits"
            ]
            lines = subprocess.check_output(cmd, encoding="utf-8").splitlines()
            
            for line in lines:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 3:
                    idx_str, used_str, total_str = parts[0], parts[1], parts[2]
                    gpu_name = parts[3] if len(parts) > 3 else "Unknown"
                    
                    idx = int(idx_str)
                    used = float(used_str)
                    total = float(total_str)
                    frac = max(0.0, min(used / total, 1.0))
                    filled = int(frac * bar_length)
                    empty = bar_length - filled

                    # choose color based on usage fraction
                    if frac < 0.5:
                        color = GREEN
                    elif frac < 0.8:
                        color = YELLOW
                    else:
                        color = RED

                    bar = f"{color}{'|' * filled}{GREY}{'-' * empty}{RESET}"
                    print(f"{EMOJI['bar']} [CUDA {idx}] {gpu_name}")
                    print(f"    {bar} {used:.0f}/{total:.0f} MiB ({frac*100:.1f}%)")
                    
        except Exception as e:
            # Fallback to PyTorch CUDA memory info
            print(f"{YELLOW}nvidia-smi not available, using PyTorch CUDA info{RESET}")
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    # Get current memory usage
                    torch.cuda.set_device(i)
                    allocated = torch.cuda.memory_allocated(i)
                    cached = torch.cuda.memory_reserved(i)
                    total = props.total_memory
                    
                    # Use allocated memory for the bar
                    used = allocated
                    frac = max(0.0, min(used / total, 1.0))
                    filled = int(frac * bar_length)
                    empty = bar_length - filled

                    if frac < 0.5:
                        color = GREEN
                    elif frac < 0.8:
                        color = YELLOW
                    else:
                        color = RED

                    bar = f"{color}{'|' * filled}{GREY}{'-' * empty}{RESET}"
                    print(f"{EMOJI['bar']} [CUDA {i}] {props.name}")
                    print(f"    {bar} {used/1024**2:.0f}/{total/1024**2:.0f} MiB ({frac*100:.1f}%)")
                    print(f"    Allocated: {allocated/1024**2:.0f} MiB, Cached: {cached/1024**2:.0f} MiB")
            except Exception as cuda_e:
                print(f"{RED}Could not get CUDA memory info: {cuda_e}{RESET}")

    # 2. Apple Silicon MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if gpu_found:
            print()  # Add spacing between different GPU types
        print(f"{BLUE}Apple Silicon MPS detected:{RESET}")
        gpu_found = True
        
        try:
            # MPS doesn't provide detailed memory info, so we show a simplified view
            # Try to get some basic info
            device = torch.device("mps")
            
            # Create a small tensor to test MPS
            test_tensor = torch.randn(100, 100, device=device)
            del test_tensor
            
            # MPS doesn't have detailed memory APIs like CUDA
            # So we'll show a basic status
            bar = f"{GREEN}{'|' * (bar_length//2)}{GREY}{'-' * (bar_length//2)}{RESET}"
            print(f"{EMOJI['bar']} [MPS] Apple Silicon GPU")
            print(f"    {bar} Status: Available (detailed memory info not supported)")
            
        except Exception as e:
            print(f"{RED}Could not access MPS: {e}{RESET}")

    # 3. AMD ROCm (if available)
    try:
        # Check if ROCm is available
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            if gpu_found:
                print()
            print(f"{BLUE}AMD ROCm GPUs detected:{RESET}")
            gpu_found = True
            
            try:
                # Try to use rocm-smi if available
                cmd = ["rocm-smi", "--showmeminfo", "vram"]
                lines = subprocess.check_output(cmd, encoding="utf-8").splitlines()
                
                # Parse rocm-smi output (this is a simplified parser)
                for i, line in enumerate(lines):
                    if "GPU" in line and "vram" in line.lower():
                        # This is a basic parser - might need adjustment based on actual output
                        bar = f"{GREEN}{'|' * (bar_length//2)}{GREY}{'-' * (bar_length//2)}{RESET}"
                        print(f"{EMOJI['bar']} [ROCm {i}] AMD GPU")
                        print(f"    {bar} (rocm-smi info available)")
                        
            except Exception:
                # Fallback for ROCm without rocm-smi
                bar = f"{GREEN}{'|' * (bar_length//2)}{GREY}{'-' * (bar_length//2)}{RESET}"
                print(f"{EMOJI['bar']} [ROCm] AMD GPU")
                print(f"    {bar} Status: Available (install rocm-smi for detailed info)")
                
    except Exception:
        pass  # ROCm not available

    # 4. Intel GPU (if available)
    try:
        # Intel GPUs are less common in PyTorch, but check anyway
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            if gpu_found:
                print()
            print(f"{BLUE}Intel XPU detected:{RESET}")
            gpu_found = True
            
            try:
                device_count = torch.xpu.device_count()
                for i in range(device_count):
                    bar = f"{GREEN}{'|' * (bar_length//2)}{GREY}{'-' * (bar_length//2)}{RESET}"
                    print(f"{EMOJI['bar']} [XPU {i}] Intel GPU")
                    print(f"    {bar} Status: Available")
            except Exception as e:
                print(f"{RED}Could not get Intel XPU info: {e}{RESET}")
                
    except Exception:
        pass  # Intel XPU not available

    # 5. Generic GPU detection fallback
    if not gpu_found:
        print(f"{RED}No supported GPU devices found.{RESET}")
        print(f"{GREY}Supported: NVIDIA CUDA, Apple MPS, AMD ROCm, Intel XPU{RESET}")
        
        # Try to detect if any GPU hardware exists but drivers are missing
        try:
            # Try lspci on Linux/macOS to detect GPU hardware
            cmd = ["lspci"]
            output = subprocess.check_output(cmd, encoding="utf-8")
            gpu_lines = [line for line in output.splitlines() if any(keyword in line.lower() for keyword in ['vga', 'display', 'nvidia', 'amd', 'intel', 'gpu'])]
            
            if gpu_lines:
                print(f"{YELLOW}GPU hardware detected but no supported drivers:{RESET}")
                for line in gpu_lines[:3]:  # Show first 3 GPUs
                    print(f"  • {line.strip()}")
                    
        except Exception:
            pass  # lspci not available or failed

def print_gpu_usage_simple():
    """
    Simple GPU usage display without color bars - useful for non-NVIDIA GPUs.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Prints available GPU backend status and memory usage when supported.
    """
    if torch is None:
        print("PyTorch not available for GPU monitoring.")
        return
        
    gpu_info = []
    
    # CUDA
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            total = props.total_memory
            gpu_info.append({
                'type': 'CUDA',
                'id': i,
                'name': props.name,
                'memory_used': allocated,
                'memory_total': total,
                'utilization': allocated / total * 100
            })
    
    # MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpu_info.append({
            'type': 'MPS',
            'id': 0,
            'name': 'Apple Silicon',
            'memory_used': None,
            'memory_total': None,
            'utilization': None
        })
    
    # ROCm
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        gpu_info.append({
            'type': 'ROCm',
            'id': 0,
            'name': 'AMD GPU',
            'memory_used': None,
            'memory_total': None,
            'utilization': None
        })
    
    # Intel XPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            gpu_info.append({
                'type': 'XPU',
                'id': i,
                'name': 'Intel GPU',
                'memory_used': None,
                'memory_total': None,
                'utilization': None
            })
    
    if gpu_info:
        print("Available GPUs:")
        for gpu in gpu_info:
            if gpu['memory_used'] is not None:
                print(f"  [{gpu['type']} {gpu['id']}] {gpu['name']}: {gpu['memory_used']/1024**2:.0f}/{gpu['memory_total']/1024**2:.0f} MiB ({gpu['utilization']:.1f}%)")
            else:
                print(f"  [{gpu['type']} {gpu['id']}] {gpu['name']}: Available")
    else:
        print("No GPUs detected.")

EMOJI = {
    "start":        "🔍",  # start
    "cpu":          "🖥️",  # CPU mode
    "mixed":        "⚙️",  # mixed CPU/GPU mode
    "gpu":          "🚀",  # RAPIDS GPU mode
    "done":         "✅",  # done
    "error":        "❌",  # error
    "bar":          "📊",  # usage bar
    "check_mark":   "✅",  # check mark
    "warning":      "⚠️",  # warning
}

class Colors:
    """ANSI color codes for terminal output styling."""
    HEADER = '\033[95m'     # Purple
    BLUE = '\033[94m'       # Blue
    CYAN = '\033[96m'       # Cyan
    GREEN = '\033[92m'      # Green
    WARNING = '\033[93m'    # Yellow
    FAIL = '\033[91m'       # Red
    ENDC = '\033[0m'        # Reset
    BOLD = '\033[1m'        # Bold
    UNDERLINE = '\033[4m'   # Underline

# Convenience function for users to check acceleration status
def check_gpu_acceleration():
    """
    Convenience function to check GPU acceleration status and get installation recommendations.
    
    This function provides a user-friendly way to check which acceleration packages
    are installed and get recommendations for optimal performance.
    
    Examples
    --------
    >>> import omicverse as ov
    >>> ov.settings.check_gpu_acceleration()
    
    Returns
    -------
    dict
        Dictionary containing installation status and recommendations
    """
    print_acceleration_status(verbose=True)
    return check_acceleration_packages()


settings = omicverseConfig()
        
