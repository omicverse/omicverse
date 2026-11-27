"""The optional radiomics facade stays lazy and forwards supported API objects."""

from __future__ import annotations

import importlib.util
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "omicverse" / "radiomics" / "__init__.py"


@pytest.fixture
def adapter():
    spec = importlib.util.spec_from_file_location("radiomics_under_test", ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plain_import_needs_only_standard_library(tmp_path):
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys

        sys.path.insert(0, sys.argv[1])
        forbidden = {
            "ov_radiomics", "SimpleITK", "radiomics", "pydicom", "nibabel",
        }

        class BlockImaging(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in forbidden:
                    raise AssertionError("Unexpected imaging import: " + fullname)

        sys.meta_path.insert(0, BlockImaging())
        import omicverse as ov
        assert "omicverse.radiomics" not in sys.modules
        facade = ov.radiomics
        import omicverse.radiomics as radiomics

        assert facade is radiomics
        assert "radiomics" in ov.__all__
        assert "radiomics" in dir(ov)
        assert set(radiomics._EXPORTS) <= set(dir(radiomics))
        assert "ov_radiomics" not in sys.modules
        print("Standard-library-only import did not load imaging packages.")
        """
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_real_missing_backend_in_standard_library_process(tmp_path):
    code = textwrap.dedent(
        """
        import sys

        sys.path.insert(0, sys.argv[1])
        import omicverse.radiomics as radiomics

        assert "ov_radiomics" not in sys.modules
        try:
            radiomics.ImageSample
        except ImportError as exc:
            assert "pip install 'omicverse[radiomics]'" in str(exc)
            assert isinstance(exc.__cause__, ModuleNotFoundError)
        else:
            raise AssertionError("A process without site-packages loaded a backend.")
        """
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code, str(ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_attribute_and_dir_do_not_load_backend(adapter, monkeypatch):
    def unexpected():
        raise AssertionError("Unknown attributes must not load a backend.")

    monkeypatch.setattr(adapter, "_load", unexpected)
    with pytest.raises(AttributeError, match="unknown_feature"):
        getattr(adapter, "unknown_feature")
    assert set(adapter._EXPORTS) <= set(dir(adapter))


@pytest.mark.parametrize(
    "cause",
    [
        ModuleNotFoundError("Missing optional package.", name="ov_radiomics"),
        ImportError("Simulated incompatible native extension."),
    ],
)
def test_unavailable_backend_explains_optional_install(adapter, monkeypatch, cause):
    def unavailable(name):
        assert name == "ov_radiomics"
        raise cause

    monkeypatch.setattr(adapter, "import_module", unavailable)
    with pytest.raises(
        ImportError, match="pip install 'omicverse\\[radiomics\\]'"
    ) as error:
        getattr(adapter, "ImageSample")
    assert error.value.__cause__ is cause


@pytest.mark.parametrize("version", ["0.1.0", "0.1.9", "0.1.0+candidate"])
def test_supported_backend_forwards_explicit_exports(adapter, monkeypatch, version):
    backend = ModuleType("ov_radiomics")
    backend.__version__ = version
    for name in adapter._EXPORTS:
        setattr(backend, name, object())
    monkeypatch.setattr(adapter, "import_module", lambda name: backend)
    assert isinstance(adapter._EXPORTS, tuple)
    assert adapter.__all__ == list(adapter._EXPORTS)
    for name in adapter._EXPORTS:
        assert getattr(adapter, name) is getattr(backend, name)


@pytest.mark.parametrize(
    "version", ["0.0.9", "0.2.0", "1.0.0", "0.1.0rc1", "bad", None]
)
def test_incompatible_version_is_explicit(adapter, monkeypatch, version):
    backend = ModuleType("ov_radiomics")
    backend.__version__ = version
    monkeypatch.setattr(adapter, "import_module", lambda name: backend)
    with pytest.raises(ImportError, match="ov-radiomics>=0.1.0,<0.2") as error:
        getattr(adapter, "extract")
    assert "pip install 'omicverse[radiomics]'" in str(error.value)


def test_every_export_and_interop_function_exists_in_installed_package():
    backend = pytest.importorskip("ov_radiomics")
    from omicverse import radiomics

    for name in radiomics._EXPORTS:
        assert getattr(radiomics, name) is getattr(backend, name)
    for name in ("from_arrays", "from_anndata", "snapshot_wsidata", "from_wsidata"):
        assert callable(getattr(backend.interop, name))
        assert getattr(radiomics.interop, name) is getattr(backend.interop, name)


def test_small_array_tutorial_matches_independent_package_exactly():
    backend = pytest.importorskip("ov_radiomics")
    from pandas.testing import assert_frame_equal

    from omicverse import radiomics

    tutorial = runpy.run_path(str(ROOT / "examples" / "radiomics_optional.py"))
    direct = tutorial["extract_small_array"](backend)
    adapted = tutorial["extract_small_array"](radiomics)
    assert_frame_equal(direct, adapted, check_exact=True)
