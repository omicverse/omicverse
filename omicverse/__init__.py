r"""
OmicVerse: A comprehensive omic framework for multi-omic analysis.

OmicVerse is a Python package that provides a unified framework for analyzing
multi-omic data including single-cell RNA-seq, bulk RNA-seq, spatial transcriptomics,
ATAC-seq, and multi-modal integration. It offers streamlined workflows for
preprocessing, analysis, visualization, and interpretation of complex biological data.

Main modules:
    bulk: Bulk RNA-seq analysis including differential expression, pathway analysis
    single: Single-cell RNA-seq analysis including clustering, trajectory inference
    bulk2single: Deconvolution and mapping between bulk and single-cell data
    space: Spatial transcriptomics analysis and integration
    pp: Preprocessing utilities for quality control and normalization
    pl: Comprehensive plotting and visualization functions
    utils: Utility functions for data handling and analysis

Key features:
    - Unified API for multiple omics data types
    - GPU acceleration support for large-scale analysis
    - Extensive visualization capabilities
    - Integration with popular bioinformatics tools
    - Comprehensive documentation and tutorials

Examples:
    >>> import omicverse as ov
    >>> adata = ov.read('data.h5ad')
    >>>
    >>> # Traditional approach
    >>> ov.pp.preprocess(adata)
    >>> ov.single.leiden(adata)
    >>> ov.pl.umap(adata, color='leiden')
    >>>
    >>> # Smart Agent approach
    >>> agent = ov.Agent(model="gpt-4o-mini", api_key="your-key")
    >>> codex_agent = ov.Agent(model="gpt-5.3-codex", auth_mode="openai_oauth")
    >>> adata = agent.run("quality control with nUMI>500, mito<0.2", adata)
    >>> adata = agent.run("preprocess with 2000 HVGs", adata)
    >>> adata = agent.run("leiden clustering resolution=1.0", adata)
"""

# Fix PyArrow compatibility issue
# PyExtensionType was renamed to ExtensionType in newer versions
import os
import sys
import importlib
try:
    import pyarrow
    if hasattr(pyarrow, 'ExtensionType') and not hasattr(pyarrow, 'PyExtensionType'):
        pyarrow.PyExtensionType = pyarrow.ExtensionType
except ImportError:
    pass

try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from pkg_resources import get_distribution
    version = lambda name: get_distribution(name).version

try:
    from omicverse_notebook import enable_formatters
    enable_formatters()
except ImportError:
    pass

# Cache for lazy-loaded modules and attributes
_lazy_modules = {}
_lazy_attrs = {}

# Define which modules should be lazy-loaded
_LAZY_MODULES = {
    'alignment',
    'bulk',
    'single',
    'bulk2single',
    'pp',
    'space',
    'pl',
    'utils',
    'io',
    'datasets',
    'es',
    'external',
    'llm',
    'agent',
    'mcp',
    'metabol',
    'micro',
    'protein',
    'genetics',
    'report',
}

# Lazy attribute mappings: {attribute_name: (module_path, attr_name)}
_LAZY_ATTRS = {
    # From io.single
    'read': ('.io.single', 'read'),

    # From pl - plotting APIs now live there
    'palette': ('.pl', 'palette'),
    'ov_plot_set': ('.pl', 'ov_plot_set'),
    'plot_set': ('.pl', 'plot_set'),
    'style': ('.pl', 'style'),

    # From _registry
    'find_function': ('._registry', 'find_function'),
    'list_functions': ('._registry', 'list_functions'),
    'get_function_help': ('._registry', 'get_function_help'),
    'recommend_function': ('._registry', 'recommend_function'),
    'export_registry': ('._registry', 'export_registry'),
    'import_registry': ('._registry', 'import_registry'),

    # From utils.smart_agent
    'Agent': ('.utils.smart_agent', 'Agent'),
    'list_supported_models': ('.utils.smart_agent', 'list_supported_models'),

    # From utils.session_notebook_executor
    'setup_kernel_for_env': ('.utils.session_notebook_executor', 'setup_kernel_for_env'),

    # From _settings - delay to avoid circular import
    'settings': ('._settings', 'settings'),
    'generate_reference_table': ('._settings', 'generate_reference_table'),

    # Common libraries - delay these too!
    'plt': ('matplotlib.pyplot', None),
    'np': ('numpy', None),
    'pd': ('pandas', None),
    'AnnData': ('anndata', 'AnnData'),
    'concat': ('anndata', 'concat'),
}

_DEFAULT_CORE_PACKAGES = (
    "scanpy",
    "anndata",
    "numpy",
    "pandas",
    "matplotlib.pyplot",
    "scipy",
    "sklearn",
)

_DEFAULT_OMICVERSE_MODULES = (
    "pp",
    "single",
    "pl",
    "bulk",
    "space",
    "utils",
)

name = "omicverse"
try:
    __version__ = version(name)
except Exception:
    __version__ = "unknown"

# Delay settings import to avoid circular dependency
# from ._settings import settings, generate_reference_table


def __getattr__(name):
    """Lazy import modules and attributes on first access."""

    # Check if it's a lazy attribute (function or class)
    if name in _LAZY_ATTRS:
        if name in _lazy_attrs:
            return _lazy_attrs[name]

        module_path, attr_name = _LAZY_ATTRS[name]
        try:
            # If no attr_name, return the module itself (for plt, np, pd)
            if attr_name is None:
                module = importlib.import_module(module_path)
                _lazy_attrs[name] = module
                return module
            else:
                # Import from omicverse submodule
                module = importlib.import_module(module_path, package='omicverse')
                attr = getattr(module, attr_name)
                _lazy_attrs[name] = attr
                return attr
        except (ImportError, AttributeError) as e:
            raise type(e)(str(e)) from e

    # Check if it's a lazy module
    if name in _LAZY_MODULES:
        if name in _lazy_modules:
            return _lazy_modules[name]

        # Use importlib.import_module for all submodules to avoid infinite
        # recursion: `from . import x` internally calls hasattr(omicverse, x)
        # which re-triggers __getattr__(x) before the module is in __dict__.
        _lazy_modules[name] = None  # re-entry guard: return None on recursion

        if name == 'llm':
            if os.environ.get("OMICVERSE_DISABLE_LLM") == "1":
                return None
            try:
                module = importlib.import_module('.llm', package='omicverse')
                _lazy_modules[name] = module
                return module
            except ImportError:
                return None

        elif name == 'datacollect':
            try:
                module = importlib.import_module('.external.datacollect', package='omicverse')
                _lazy_modules[name] = module
                return module
            except ImportError:
                return None

        elif name == 'agent':
            try:
                module = importlib.import_module('.agent', package='omicverse')
                _lazy_modules[name] = module
                return module
            except ImportError:
                return None

        else:
            try:
                module = importlib.import_module(f'.{name}', package='omicverse')
                _lazy_modules[name] = module
                return module
            except Exception as e:
                # Catch Exception (not just ImportError) because transitive
                # imports can fail with TypeError, OSError, ValueError, etc.
                # The scipy 1.17.x _new_distributions module, for instance,
                # runs doc-generation at import time and raises TypeError
                # when torch isn't installed — that fires when sklearn pulls
                # scipy.stats in during the docs build, and was masking this
                # helper's sphinx stub.
                _lazy_modules.pop(name, None)
                # Under Sphinx builds, return an empty stub module instead of
                # raising, so autosummary/autodoc can still introspect the
                # name without the entire build aborting when an optional
                # heavy dependency (marsilea, torch, tangram, …) is absent.
                if "sphinx" in sys.modules:
                    import types
                    stub = types.ModuleType(f"omicverse.{name}")
                    stub.__doc__ = (
                        f"omicverse.{name} could not be imported in the docs "
                        f"environment: {type(e).__name__}: {e}."
                    )
                    _lazy_modules[name] = stub
                    return stub
                raise AttributeError(
                    f"Failed to import omicverse.{name}: "
                    f"{type(e).__name__}: {e}. "
                    f"A required dependency may not be installed."
                ) from e

    # If not found, raise AttributeError
    raise AttributeError(f"Module 'omicverse' has no attribute '{name}'")


def __dir__():
    """Provide a complete list of available attributes for tab completion."""
    base_attrs = [
        # Lazy attributes
        "read", "palette", "ov_plot_set", "plot_set", "style",
        "find_function", "list_functions", "get_function_help",
        "recommend_function", "export_registry", "import_registry",
        "Agent", "list_supported_models", "setup_kernel_for_env",

        # Settings
        "settings", "generate_reference_table",

        # Common libraries
        "plt", "np", "pd", "AnnData", "concat",

        # Version
        "__version__",
        "load_package",
    ]
    # Add utils to lazy modules for proper handling
    lazy_modules_with_utils = _LAZY_MODULES | {'utils', 'datacollect'}
    return sorted(set(base_attrs + list(lazy_modules_with_utils)))


def load_package(
    packages=None,
    omicverse_modules=None,
    include_omicverse_modules=True,
    show_summary=True,
):
    """
    Preload core dependencies to warm up lazy imports.

    Parameters
    ----------
    packages : list[str] | tuple[str] | None
        External package import paths. Defaults to common heavy dependencies
        such as scanpy and torch.
    omicverse_modules : list[str] | tuple[str] | None
        Top-level omicverse modules to preload via lazy getattr.
    include_omicverse_modules : bool
        Whether to preload omicverse lazy modules in addition to external
        dependencies.
    show_summary : bool
        Whether to print a short success/failure summary at the end.

    Returns
    -------
    dict
        Dict containing `loaded`, `failed`, and `results`.
    """
    package_list = tuple(packages) if packages is not None else _DEFAULT_CORE_PACKAGES
    module_list = tuple(omicverse_modules) if omicverse_modules is not None else _DEFAULT_OMICVERSE_MODULES

    tasks = [("package", p) for p in package_list]
    if include_omicverse_modules:
        tasks.extend(("omicverse", m) for m in module_list)

    try:
        from tqdm.auto import tqdm
        iterator = tqdm(tasks, desc="ov.load_package", unit="item")
    except Exception:
        tqdm = None
        iterator = tasks

    results = []
    current_module = sys.modules[__name__]

    for task_type, target in iterator:
        error = None
        try:
            if task_type == "package":
                importlib.import_module(target)
            else:
                getattr(current_module, target)
        except Exception as exc:
            error = str(exc)

        success = error is None
        results.append(
            {
                "type": task_type,
                "name": target,
                "success": success,
                "error": error,
            }
        )
        if tqdm is not None:
            status = "OK" if success else "FAIL"
            iterator.set_postfix_str(f"{status}: {target}")

    loaded = [r["name"] for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if show_summary:
        print(f"ov.load_package loaded {len(loaded)}/{len(results)} targets.")
        if failed:
            failed_names = ", ".join(f"{r['type']}:{r['name']}" for r in failed)
            print(f"Failed targets: {failed_names}")

    return {"loaded": loaded, "failed": failed, "results": results}


__all__ = [
    # Lazy-loaded modules
    "alignment",
    "bulk",
    "single",
    "utils",
    "bulk2single",
    "pp",
    "space",
    "pl",
    "metabol",
    "protein",
    "genetics",
    "datasets",
    "external",
    "llm",
    "datacollect",
    "agent",

    # Essential utilities (lazy-loaded)
    "read",
    "palette",
    "ov_plot_set",
    "plot_set",
    "style",

    # Function registry (lazy-loaded)
    "find_function",
    "list_functions",
    "get_function_help",
    "recommend_function",
    "export_registry",
    "import_registry",

    # Smart Agent (lazy-loaded)
    "Agent",
    "list_supported_models",
    "setup_kernel_for_env",

    # Settings
    "settings",
    "generate_reference_table",

    # Common libraries (lazy-loaded)
    "plt",
    "np",
    "pd",
    "AnnData",
    "concat",
    "load_package",

    # Version
    "__version__",
]
