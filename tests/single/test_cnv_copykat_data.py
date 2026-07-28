"""``method='copykat'`` guard for the pycopykat genome reference tables.

pycopykat up to 0.1.0.dev1 kept ``hg20_gene_anno.parquet`` at its repo root
and resolved it one directory above the package, so a wheel install looked in
``site-packages/data/`` — a directory that does not exist. The failure only
surfaced deep inside ``pandas.read_parquet`` (omicverse#903); the guard turns
it into an ImportError naming the reinstall command.
"""
from __future__ import annotations

import sys
import types

import pytest

ov = pytest.importorskip("omicverse")

from omicverse.single._cnv import _check_copykat_data  # noqa: E402


def _fake_pycopykat(monkeypatch, pkg_dir):
    mod = types.ModuleType("pycopykat")
    mod.__file__ = str(pkg_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "pycopykat", mod)


def test_accepts_data_inside_the_package(monkeypatch, tmp_path):
    pkg = tmp_path / "site-packages" / "pycopykat"
    (pkg / "data").mkdir(parents=True)
    _fake_pycopykat(monkeypatch, pkg)
    _check_copykat_data()  # must not raise


def test_accepts_the_legacy_source_checkout_layout(monkeypatch, tmp_path):
    # Editable install: <repo>/pycopykat/ with the tables at <repo>/data/.
    pkg = tmp_path / "py-CopyKAT" / "pycopykat"
    pkg.mkdir(parents=True)
    (pkg.parent / "data").mkdir()
    _fake_pycopykat(monkeypatch, pkg)
    _check_copykat_data()  # must not raise


def test_rejects_an_install_without_the_tables(monkeypatch, tmp_path):
    pkg = tmp_path / "site-packages" / "pycopykat"
    pkg.mkdir(parents=True)
    _fake_pycopykat(monkeypatch, pkg)
    with pytest.raises(ImportError, match="genome reference tables"):
        _check_copykat_data()


def test_error_names_the_reinstall_command(monkeypatch, tmp_path):
    pkg = tmp_path / "site-packages" / "pycopykat"
    pkg.mkdir(parents=True)
    _fake_pycopykat(monkeypatch, pkg)
    with pytest.raises(ImportError) as exc:
        _check_copykat_data()
    assert "py-CopyKAT.git" in str(exc.value)
