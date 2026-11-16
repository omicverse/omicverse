"""An accepted local wheel does not activate a public-registry installation promise."""

import ast
import importlib.util
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_unpublished_extra_stays_a_commented_placeholder():
    metadata_path = ROOT / "pyproject.toml"
    metadata = tomllib.loads(metadata_path.read_text(encoding="utf-8"))
    assert "radiomics" not in metadata["project"]["optional-dependencies"]
    source = metadata_path.read_text(encoding="utf-8")
    assert '# radiomics = ["ov-radiomics>=0.1.0,<0.2"]' in source
    assert "Enable only after ov-radiomics is publicly installable" in source


def test_missing_backend_guidance_gates_future_extra(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "radiomics_installation_gate", ROOT / "omicverse/radiomics/__init__.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    def missing(name):
        raise ModuleNotFoundError("Backend is absent.", name=name)

    monkeypatch.setattr(adapter, "import_module", missing)
    with pytest.raises(ImportError) as error:
        adapter.ImageSample
    message = str(error.value)
    assert "accepted local wheel" in message
    assert "<path-to-accepted-ov-radiomics-wheel.whl>" in message
    assert message.index("After registry publication") < message.index(
        "pip install 'omicverse[radiomics]'"
    )


def test_tutorial_marks_extra_command_as_future_registry_installation():
    source = (ROOT / "examples/radiomics_optional.py").read_text(encoding="utf-8")
    guidance = ast.get_docstring(ast.parse(source))
    assert "accepted local wheel" in guidance
    assert "commented metadata placeholder" in guidance
    assert guidance.index("After registry") < guidance.index(
        "pip install 'omicverse[radiomics]'"
    )
