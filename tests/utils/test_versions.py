"""Generic package-version helpers in ``omicverse.utils._versions``.

These gate optional backends behind a minimum version (e.g. pyscdblfinder
>=0.2.0, see issue #848) and are reused across omicverse, so they must behave
predictably for installed / missing / older / newer packages.
"""
import pytest

from omicverse.utils import _versions
from omicverse.utils import installed_version, version_lt, version_at_least


def test_public_reexports_match_module():
    assert installed_version is _versions.installed_version
    assert version_lt is _versions.version_lt
    assert version_at_least is _versions.version_at_least


def test_installed_version_known_and_unknown():
    # pytest is always installed in the test env
    assert isinstance(installed_version("pytest"), str)
    # a package that cannot exist
    assert installed_version("omicverse-no-such-dist-xyz") is None


def test_version_lt_ordering():
    assert version_lt("0.1.0", "0.2.0")
    assert version_lt("0.1.9", "0.2.0")
    assert version_lt("1.9.0", "1.10.0")   # numeric, not lexical
    assert not version_lt("0.2.0", "0.2.0")
    assert not version_lt("0.2.1", "0.2.0")
    assert not version_lt("1.0.0", "0.2.0")


def test_version_at_least(monkeypatch):
    monkeypatch.setattr(_versions, "installed_version", lambda pkg: "0.1.0")
    ok, ver = version_at_least("anything", "0.2.0")
    assert ok is False and ver == "0.1.0"

    monkeypatch.setattr(_versions, "installed_version", lambda pkg: "0.2.0")
    ok, ver = version_at_least("anything", "0.2.0")
    assert ok is True and ver == "0.2.0"

    monkeypatch.setattr(_versions, "installed_version", lambda pkg: None)
    ok, ver = version_at_least("anything", "0.2.0")
    assert ok is False and ver is None
