"""``_resolve_doublets_method`` must not let an outdated pyscdblfinder stall the
pipeline on large datasets.

pyscdblfinder <0.2.0 builds a full N×N distance matrix in its kNN step, which
needs ~N²·8 bytes (≈125 GB at 100k cells) and thrashes swap — near-zero CPU for
hours, never finishing (issue #848). When such a version is installed we fall
back to 'scrublet' with an upgrade hint so the run completes; 0.2.0+ passes
through. The version is read from the *distribution* metadata because 0.2.0
ships a stale ``__version__ = '0.1.0'`` in its ``__init__``.
"""
import sys
import types

import pytest

from omicverse.pp import _qc


@pytest.fixture
def fake_pyscdblfinder(monkeypatch):
    """Install a dummy ``pyscdblfinder`` module so the import succeeds, and let
    tests control what ``importlib.metadata`` reports for its version."""
    mod = types.ModuleType("pyscdblfinder")
    mod.__version__ = "0.1.0"  # deliberately stale, must be ignored
    monkeypatch.setitem(sys.modules, "pyscdblfinder", mod)

    def _set_dist_version(v):
        monkeypatch.setattr(_qc, "_installed_version",
                            lambda pkg: v if pkg == "pyscdblfinder" else None)

    return _set_dist_version


def test_outdated_version_falls_back_to_scrublet(fake_pyscdblfinder, capsys):
    fake_pyscdblfinder("0.1.0")
    assert _qc._resolve_doublets_method("scdblfinder") == "scrublet"
    out = capsys.readouterr().out
    assert "848" in out and "0.2.0" in out


def test_current_version_passes_through(fake_pyscdblfinder):
    fake_pyscdblfinder("0.2.0")
    assert _qc._resolve_doublets_method("scdblfinder") == "scdblfinder"


def test_newer_version_passes_through(fake_pyscdblfinder):
    fake_pyscdblfinder("0.3.1")
    assert _qc._resolve_doublets_method("scdblfinder") == "scdblfinder"


def test_missing_package_falls_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyscdblfinder", None)  # forces ImportError
    assert _qc._resolve_doublets_method("scdblfinder") == "scrublet"


def test_other_methods_unchanged():
    for m in ("scrublet", "sccomposite", "doubletfinder"):
        assert _qc._resolve_doublets_method(m) == m


def test_version_helpers():
    assert _qc._version_lt("0.1.0", "0.2.0")
    assert _qc._version_lt("0.1.9", "0.2.0")
    assert not _qc._version_lt("0.2.0", "0.2.0")
    assert not _qc._version_lt("0.2.1", "0.2.0")
    assert not _qc._version_lt("1.0.0", "0.2.0")
