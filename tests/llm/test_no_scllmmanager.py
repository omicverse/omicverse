"""SCLLMManager was removed; ModelFactory is the supported entry point."""
import importlib


def test_scllmmanager_removed():
    ov = importlib.import_module("omicverse")
    assert not hasattr(ov.llm, "SCLLMManager")
    assert "SCLLMManager" not in getattr(ov.llm, "__all__", [])
    mf = importlib.import_module("omicverse.llm.model_factory")
    assert not hasattr(mf, "SCLLMManager")


def test_model_factory_lists_models():
    pytest = importlib.import_module("pytest")
    pytest.importorskip("torch")
    ov = importlib.import_module("omicverse")
    names = set(ov.llm.ModelFactory.available_models())
    assert {"scgpt", "scfoundation", "geneformer", "cellplm", "uce"} <= names
