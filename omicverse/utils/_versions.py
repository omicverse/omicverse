"""Generic package-version helpers.

Small, dependency-light utilities for querying and comparing installed package
versions. Use these to gate an optional backend behind a minimum version — e.g.
so an outdated wheel with a known-bad code path can't silently degrade or stall
a pipeline. Reusable anywhere in omicverse an optional dependency has a
known-bad older release (pyscdblfinder, pydoubletfinder, ...).

Versions are read from the *distribution* metadata via ``importlib.metadata``
rather than a module's ``__version__`` attribute, which can be stale or missing
(e.g. pyscdblfinder 0.2.0 ships ``__version__ = '0.1.0'``).
"""
from __future__ import annotations

from typing import Optional, Tuple

__all__ = ["installed_version", "version_lt", "version_at_least"]


def installed_version(pkg: str) -> Optional[str]:
    """Return the installed distribution version of ``pkg`` (or ``None``).

    Parameters
    ----------
    pkg
        Distribution name as known to pip (e.g. ``"pyscdblfinder"``).

    Returns
    -------
    str or None
        The version string from distribution metadata, or ``None`` when the
        package is not installed.
    """
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # pragma: no cover - py<3.8
        from importlib_metadata import version, PackageNotFoundError  # type: ignore
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def version_lt(have, ref) -> bool:
    """Return ``True`` when version string ``have`` is older than ``ref``."""
    try:
        from packaging.version import parse
        return parse(str(have)) < parse(str(ref))
    except Exception:  # pragma: no cover - packaging ships with scanpy/anndata
        def _tup(s):
            out = []
            for part in str(s).split(".")[:3]:
                digits = "".join(ch for ch in part if ch.isdigit())
                out.append(int(digits) if digits else 0)
            return tuple(out)
        return _tup(have) < _tup(ref)


def version_at_least(pkg: str, min_version: str) -> Tuple[bool, Optional[str]]:
    """Check whether an installed distribution meets a minimum version.

    Parameters
    ----------
    pkg
        Distribution name (e.g. ``"pyscdblfinder"``).
    min_version
        Minimum acceptable version (inclusive), e.g. ``"0.2.0"``.

    Returns
    -------
    (ok, version) : tuple[bool, str | None]
        ``ok`` is ``True`` only when ``pkg`` is installed **and** its version
        is ``>= min_version``. ``version`` is the installed version string, or
        ``None`` when the package is not installed (in which case ``ok`` is
        ``False``).
    """
    ver = installed_version(pkg)
    if ver is None:
        return False, None
    return (not version_lt(ver, min_version)), ver
