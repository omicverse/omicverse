r"""
External algorithms and specialized methods collection.

This module provides access to a comprehensive collection of external algorithms
and specialized methods that extend omicverse's core functionality. These include
state-of-the-art methods for spatial transcriptomics, trajectory inference,
multi-omics integration, and advanced single-cell analysis.

Algorithm categories:
    Spatial transcriptomics:
        - STAGATE_pyG: Spatial domain identification
        - STAligner: Multi-sample spatial alignment
        - GraphST: Graph-based spatial analysis
        - PROST: Pattern recognition for spatial data
        - spatrio: Spatial transcriptomics analysis
        - CAST: Cellular spatial organization
        - starfysh: Spatial factorization
        
    Trajectory inference:
        - STT: Spatial transition tensor
        - VIA: Velocity integration and annotation
        - bonsai: Maximum-likelihood cell-state tree (CC BY-NC 4.0)
        - cytotrace2: Developmental potential scoring
        - monocle2_py: Pure-Python reimplementation of Monocle 2 (DDRTree)
        
    Multi-omics integration:
        - mofapy2: Multi-Omics Factor Analysis
        - GNTD: Graph-regularized non-negative tensor decomposition
        - scMulan: Multi-modal analysis
        
    Single-cell methods:
        - tosica: Trajectory inference and cell fate
        - cnmf: Consensus non-negative matrix factorization
        - CEFCON: Cell fate controller networks
        - cellanova: Cell type-aware analysis
        - BINARY: Binary classification methods
        - flowsig: Flow cytometry-style analysis
        - scdiffusion: Diffusion-based methods
        
    Network analysis:
        - PyWGCNA: Weighted gene co-expression networks
        - commot: Cell-cell communication
        
    Data integration:
        - scanorama: Batch correction and integration
        - scSLAT: Single-cell alignment
        
    Spatial methods:
        - spaceflow: Spatial flow analysis
        - gaston: Spatial depth estimation

Features:
    - Optional dependencies: Algorithms are loaded as needed
    - Consistent interfaces: Wrapped for omicverse compatibility
    - Performance optimized: GPU acceleration where available
    - Well documented: Each method includes comprehensive examples

Notes:
    - Some algorithms require additional dependencies
    - GPU methods require appropriate hardware and drivers
    - All methods are thoroughly tested and validated
    - Regular updates incorporate latest algorithmic advances
"""

from .._optional import build_optional_dependency_error, missing_optional_dependency

# Cache for lazy-loaded modules
_lazy_modules = {}

_TORCH_HEAVY_MODULES = {
    'scSLAT',
    'CEFCON',
    'GNTD',
    'spaceflow',
    'tosica',
    'STAGATE_pyG',
    'STAligner',
    'PROST',
    'cytotrace2',
    'GraphST',
    'starfysh',
    'scdiffusion',
    'BINARY',
    'gaston',
    'nocd',
    'popv',
}

__all__ = [
    'bonsai',
    'scSLAT',
    'CEFCON',
    'mofapy2',
    'GNTD',
    'spaceflow',
    'STT',
    'tosica',
    'STAGATE_pyG',
    'STAligner',
    'spatrio',
    'PROST',
    'cytotrace2',
    'GraphST',
    'commot',
    'cnmf',
    'starfysh',
    #'scMulan',
    'flowsig',
    'PyWGCNA',
    'CAST',
    'scanorama',
    'scdiffusion',
    'BINARY',
    'cellanova',
    'VIA',
    'gaston',
    'pyscenic',
    'bin2cell',
    'sude_py',
    'harmony',
    'datacollect',
    'spaco',
    'nocd',
    'popv',
    'SpatialDE',
    'NaiveDE',
    'somde'
    ,
    'cellcharter',
    'monocle2_py',
    'gsmap',
]


def __getattr__(name):
    """
    Lazy import modules on first access.
    
    This function is called when an attribute is accessed that doesn't exist
    in the module's namespace. It dynamically imports the requested module
    only when it's actually needed, avoiding the overhead of importing all
    external modules at once.
    
    Parameters
    ----------
    name : str
        The name of the module to import
        
    Returns
    -------
    module
        The imported module
        
    Raises
    ------
    AttributeError
        If the requested module is not available or not in __all__
    """
    if name not in __all__:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'"
        )
    
    # Check cache first
    if name in _lazy_modules:
        return _lazy_modules[name]
    
    # Import the module dynamically
    try:
        import importlib
        module = importlib.import_module(f'.{name}', package='omicverse.external')
        _lazy_modules[name] = module
        return module
    except ImportError as e:
        if name in _TORCH_HEAVY_MODULES and missing_optional_dependency(e, ("torch", "torch_geometric")):
            raise build_optional_dependency_error(
                f"omicverse.external.{name}",
                ("torch", "torch_geometric"),
            ) from e
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. "
            f"Failed to import: {e}"
        ) from e


def __dir__():
    """
    Provide a complete list of available attributes for tab completion.
    
    Returns
    -------
    list
        List of available module names
    """
    return __all__
