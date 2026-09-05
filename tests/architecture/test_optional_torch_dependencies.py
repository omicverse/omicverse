import builtins
import importlib
import sys
import types

import pytest


_MANAGED_PREFIXES = ("omicverse", "torch", "torch_geometric")


def _matching_module_names(prefixes):
    return [
        name for name in list(sys.modules)
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    ]


@pytest.fixture(autouse=True)
def _restore_managed_modules():
    saved = {name: sys.modules.get(name) for name in _matching_module_names(_MANAGED_PREFIXES)}
    yield
    for name in _matching_module_names(_MANAGED_PREFIXES):
        sys.modules.pop(name, None)
    for name, module in saved.items():
        if module is not None:
            sys.modules[name] = module


def _clear_modules(prefixes):
    for name in _matching_module_names(prefixes):
        sys.modules.pop(name, None)


def _block_torch_imports(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root in {"torch", "torch_geometric"}:
            raise ImportError(f"blocked import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _install_settings_stub():
    settings_module = types.ModuleType("omicverse._settings")
    settings_module.add_reference = lambda *args, **kwargs: None
    settings_module.settings = object()

    class _DummyColors:
        GREEN = ""
        RED = ""
        YELLOW = ""
        BLUE = ""
        END = ""

    def _fallback(name):
        if name == "Colors":
            return _DummyColors
        if name == "EMOJI":
            return {}
        return lambda *args, **kwargs: None

    settings_module.__getattr__ = _fallback
    sys.modules["omicverse._settings"] = settings_module


def _install_space_dependency_stubs():
    module_defs = {
        "omicverse.space._tangram": {"Tangram": lambda *args, **kwargs: None},
        "omicverse.space._spatrio": {
            "CellMap": lambda *args, **kwargs: None,
            "CellLoc": lambda *args, **kwargs: None,
        },
        "omicverse.space._stt": {"STT": lambda *args, **kwargs: None},
        "omicverse.space._svg": {
            "svg": lambda *args, **kwargs: None,
            "spatial_neighbors": lambda *args, **kwargs: None,
            "spatial_autocorr": lambda *args, **kwargs: None,
            "moranI": lambda *args, **kwargs: None,
        },
        "omicverse.space._cast": {"CAST": lambda *args, **kwargs: None},
        "omicverse.space._tools": {"__all__": []},
        "omicverse.space._commot": {
            "create_communication_anndata": lambda *args, **kwargs: None,
            "update_classification_from_database": lambda *args, **kwargs: None,
        },
        "omicverse.space._deconvolution": {
            "Deconvolution": lambda *args, **kwargs: None,
            "calculate_gene_signature": lambda *args, **kwargs: None,
        },
    }

    for module_name, attrs in module_defs.items():
        module = types.ModuleType(module_name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[module_name] = module


def _install_bulk2single_dependency_stubs():
    module = types.ModuleType("omicverse.bulk2single._utils")
    module.bulk2single_plot_cellprop = lambda *args, **kwargs: None
    module.bulk2single_plot_correlation = lambda *args, **kwargs: None
    module.load_data = lambda *args, **kwargs: None
    module.data_process = lambda *args, **kwargs: None
    module.bulk2single_data_prepare = lambda *args, **kwargs: None
    sys.modules["omicverse.bulk2single._utils"] = module


def _install_bulk_dependency_stubs():
    module_defs = {
        "omicverse.bulk._Enrichment": {
            "pyGSEA": object(),
            "pyGSE": object(),
            "geneset_enrichment": object(),
            "geneset_plot": object(),
            "geneset_enrichment_GSEA": object(),
            "geneset_plot_multi": object(),
            "enrichment_multi_concat": object(),
        },
        "omicverse.bulk._network": {
            "pyPPI": object(),
            "string_interaction": object(),
            "string_map": object(),
            "generate_G": object(),
        },
        "omicverse.bulk._chm13": {
            "get_chm13_gene": object(),
            "find_chm13_gene": object(),
        },
        "omicverse.bulk._Deseq2": {
            "pyDEG": object(),
            "deseq2_normalize": object(),
            "estimateSizeFactors": object(),
            "estimateDispersions": object(),
            "Matrix_ID_mapping": object(),
            "data_drop_duplicates_index": object(),
            "temporal_clusters": object(),
        },
        "omicverse.bulk._tcga": {"pyTCGA": object()},
        "omicverse.bulk._combat": {"batch_correction": object()},
    }

    for module_name, attrs in module_defs.items():
        module = types.ModuleType(module_name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[module_name] = module


def _install_single_dependency_stubs():
    module_defs = {
        "omicverse.single._cosg": {"cosg": lambda *args, **kwargs: "cosg-ok"},
        "omicverse.single._anno": {
            "pySCSA": lambda *args, **kwargs: "anno-ok",
            "MetaTiME": object(),
            "scanpy_lazy": object(),
            "scanpy_cellanno_from_dict": object(),
            "get_celltype_marker": object(),
        },
        "omicverse.single._mofa": {
            "pyMOFAART": object(),
            "pyMOFA": object(),
            "GLUE_pair": object(),
            "factor_exact": object(),
            "factor_correlation": object(),
            "get_weights": object(),
            "glue_pair": object(),
            "get_r2_from_hdf5": object(),
            "convert_r2_to_matrix": object(),
            "factor_group_correlation_mdata": object(),
            "plot_factor_group_associations": object(),
            "plot_factor_boxplots": object(),
            "plot_factors_violin": object(),
            "plot_weights": object(),
        },
        "omicverse.single._scdrug": {
            "writeGEP": object(),
            "Drug_Response": object(),
        },
        "omicverse.single._autoresolution": {
            "auto_resolution": object(),
            "autoResolution": object(),
        },
        "omicverse.single._cpdb": {
            "cpdb_network_cal": object(),
            "cpdb_plot_network": object(),
            "cpdb_plot_interaction": object(),
            "cpdb_interaction_filtered": object(),
            "cpdb_submeans_exacted": object(),
            "cpdb_exact_target": object(),
            "cpdb_exact_source": object(),
            "cellphonedb_v5": object(),
            "run_cellphonedb_v5": object(),
            "format_cpdb_results": object(),
            "format_cpdb_results_for_viz": object(),
            "download_cellphonedb_database": object(),
            "validate_cpdb_database": object(),
        },
        "omicverse.single._liana": {
            "run_liana": object(),
            "format_liana_results": object(),
            "format_liana_results_for_viz": object(),
        },
        "omicverse.single._comm": {
            "to_comm_adata": object(),
            "extract_comm_adata": object(),
        },
        "omicverse.single._scgsea": {
            "geneset_aucell": object(),
            "pathway_aucell": object(),
            "pathway_aucell_enrichment": object(),
            "geneset_aucell_tmp": object(),
            "pathway_aucell_tmp": object(),
            "pathway_aucell_enrichment_tmp": object(),
            "pathway_enrichment": object(),
            "pathway_enrichment_plot": object(),
        },
        "omicverse.single._atac": {
            "atac_concat_get_index": object(),
            "atac_concat_inner": object(),
            "atac_concat_outer": object(),
        },
        "omicverse.single._batch": {"batch_correction": object()},
        "omicverse.single._diffusionmap": {"diffmap": object()},
        "omicverse.single._aucell": {"aucell": object()},
        "omicverse.single._metacell": {
            "MetaCell": object(),
            "plot_metacells": object(),
            "get_obs_value": object(),
            "optimize_granularity": object(),
            "compare_metacell_backends": object(),
        },
        "omicverse.single._mdic3": {"pyMDIC3": object()},
        "omicverse.single._gptcelltype": {
            "gptcelltype": object(),
            "gpt4celltype": object(),
            "get_cluster_celltype": object(),
        },
        "omicverse.single._gptcelltype_local": {"gptcelltype_local": object()},
        "omicverse.single._sccaf": {
            "SCCAF_assessment": object(),
            "plot_roc": object(),
            "SCCAF_optimize_all": object(),
            "color_long": object(),
        },
        "omicverse.single._multimap": {
            "TFIDF_LSI": object(),
            "Wrapper": object(),
            "Integration": object(),
            "Batch": object(),
        },
        "omicverse.single._cellvote": {
            "get_cluster_celltype": object(),
            "CellVote": object(),
            "cellvote_consensus_score": object(),
        },
        "omicverse.single._deg_ct": {"DCT": object(), "DEG": object()},
        "omicverse.single._lazy_function": {"lazy": object()},
        "omicverse.single._lazy_report": {"generate_scRNA_report": object()},
        "omicverse.single._lazy_checkpoint": {
            "lazy_checkpoint": object(),
            "resume_from_checkpoint": object(),
            "list_checkpoints": object(),
            "cleanup_checkpoints": object(),
        },
        "omicverse.single._monocle": {"Monocle": object()},
        "omicverse.single._lazy_step_by_step": {
            "lazy_step_qc": object(),
            "lazy_step_preprocess": object(),
            "lazy_step_scale": object(),
            "lazy_step_pca": object(),
            "lazy_step_cell_cycle": object(),
            "lazy_step_harmony": object(),
            "lazy_step_scvi": object(),
            "lazy_step_select_best_method": object(),
            "lazy_step_mde": object(),
            "lazy_step_clustering": object(),
            "lazy_step_final_embeddings": object(),
            "lazy_step_by_step_guide": object(),
        },
        "omicverse.single._cellmatch": {
            "CellOntologyMapper": object(),
            "download_cl": object(),
        },
        "omicverse.single._scenic": {
            "SCENIC": object(),
            "grn": object(),
            "build_correlation_network_umap_layout": object(),
            "add_tf_regulation": object(),
            "plot_grn": object(),
        },
        "omicverse.single._annotation": {"Annotation": object()},
        "omicverse.single._annotation_ref": {"AnnotationRef": object()},
        "omicverse.single._velo": {
            "Velo": object(),
            "velocity_embedding": object(),
            "velocity": object(),
            "cellrank_fate": object(),
            "state_names": object(),
            "clean_lineages": object(),
            "perturbation_effect": object(),
            "cell_fate_perturbation": object(),
            "velocity_effect": object(),
        },
        "omicverse.single._milo_dev": {"Milo": object()},
        "omicverse.single._markers": {"find_markers": object(), "get_markers": object()},
        "omicverse.single._dynamic_features": {
            "DynamicFeaturesResult": object(),
            "dynamic_features": object(),
        },
        "omicverse.single._cefcon": {
            "pyCEFCON": object(),
            "convert_human_to_mouse_network": lambda *args, **kwargs: "convert-network-ok",
            "load_human_prior_interaction_network": lambda *args, **kwargs: "load-network-ok",
            "mouse_hsc_nestorowa16": lambda *args, **kwargs: "mouse-hsc-ok",
        },
    }

    for module_name, attrs in module_defs.items():
        module = types.ModuleType(module_name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[module_name] = module


def test_public_modules_import_without_torch(monkeypatch):
    _clear_modules(["omicverse", "torch", "torch_geometric"])
    _block_torch_imports(monkeypatch)
    _install_settings_stub()
    _install_space_dependency_stubs()
    _install_bulk2single_dependency_stubs()
    _install_bulk_dependency_stubs()

    ov = importlib.import_module("omicverse")

    assert ov is not None
    assert ov.space is not None
    assert ov.bulk is not None
    assert ov.bulk2single is not None
    assert ov.llm is not None


def test_torch_backed_features_raise_consistent_error(monkeypatch):
    _clear_modules(["omicverse", "torch", "torch_geometric"])
    _block_torch_imports(monkeypatch)
    _install_settings_stub()
    _install_space_dependency_stubs()
    _install_bulk2single_dependency_stubs()
    _install_bulk_dependency_stubs()

    ov = importlib.import_module("omicverse")

    with pytest.raises(ImportError, match="omicverse.space clustering requires the optional dependencies `torch`, `torch_geometric`"):
        ov.space.pySTAGATE(None)

    with pytest.raises(ImportError, match="omicverse.bulk2single.Bulk2Single requires the optional dependencies `torch`"):
        ov.bulk2single.Bulk2Single(None, None, "celltype")

    with pytest.raises(ImportError, match="omicverse.bulk.Deconvolution requires the optional dependencies `torch`"):
        ov.bulk.Deconvolution()

    with pytest.raises(ImportError, match="omicverse.llm requires the optional dependencies `torch`"):
        ov.llm.SCLLMManager()


def test_single_imports_without_torch_and_fails_on_torch_features(monkeypatch):
    _clear_modules(["omicverse", "torch", "torch_geometric"])
    _block_torch_imports(monkeypatch)
    _install_settings_stub()
    _install_single_dependency_stubs()

    ov = importlib.import_module("omicverse")

    assert ov.single is not None
    assert ov.single.cosg() == "cosg-ok"
    assert ov.single.mouse_hsc_nestorowa16() == "mouse-hsc-ok"
    assert ov.single.load_human_prior_interaction_network() == "load-network-ok"
    assert ov.single.convert_human_to_mouse_network() == "convert-network-ok"

    with pytest.raises(ImportError, match="omicverse.single.pySIMBA requires the optional dependencies `torch`"):
        ov.single.pySIMBA()

    with pytest.raises(ImportError, match="omicverse.single.scnocd requires the optional dependencies `torch`, `torch_geometric`"):
        ov.single.scnocd()

    with pytest.raises(ImportError, match="omicverse.single.pyCEFCON requires the optional dependencies `torch`, `torch_geometric`"):
        ov.single.pyCEFCON()
