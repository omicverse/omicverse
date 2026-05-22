r"""Lazy-import dependency gates for :mod:`omicverse.mol`.

The structural-biology stack is optional — heavy backends are imported
*inside* the functions that use them, and a missing one raises a clean,
actionable :class:`ImportError` naming the pip extra to install.
"""

from __future__ import annotations

import importlib
from typing import Any


def _need(module: str, extra: str, feature: str) -> Any:
    """Import ``module`` or raise an actionable ImportError.

    Parameters
    ----------
    module
        The importable module name (e.g. ``'py3Dmol'``).
    extra
        The omicverse pip extra that provides it (``'mol'`` / ``'mol-dock'``).
    feature
        Human-readable description of what needs it, for the error message.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the env
        raise ImportError(
            f"{feature} needs the '{module}' package, which is not installed.\n"
            f"  Install the omicverse molecular stack with:\n"
            f"    pip install 'omicverse[{extra}]'"
        ) from exc


def check_core() -> None:
    """Gate for the core ov.mol stack (structure + interactive viz)."""
    _need("biotite", "mol", "ov.mol structure handling")
    _need("py3Dmol", "mol", "ov.mol interactive visualization")


def check_dock() -> None:
    """Gate for the docking layer (rdkit / vina / meeko)."""
    _need("rdkit", "mol-dock", "ov.mol docking")
    _need("vina", "mol-dock", "ov.mol docking (AutoDock Vina)")
    _need("meeko", "mol-dock", "ov.mol docking (ligand preparation)")


def check_pocket() -> None:
    """Gate for the binding-pocket backend (rust-fpocket)."""
    try:
        importlib.import_module("fpocket_rs")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ov.mol.pockets needs the 'fpocket-rs' package (a pip-installable "
            "Rust port of fpocket), which is not installed.\n"
            "  Install it with:\n"
            "    pip install fpocket-rs"
        ) from exc
