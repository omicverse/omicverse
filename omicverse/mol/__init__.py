r"""
Molecular structure & drug-binding analysis for omicverse.

``ov.mol`` is the bridge from an omics result — a differential gene from
``ov.bulk``, a marker from ``ov.single``, a variant from ``ov.genetics`` —
to the **3D structure** of the target protein and its **drug context**:
*what does the target look like, and can it be drugged?*

Quick-start
-----------
>>> import omicverse as ov
>>> s = ov.mol.fetch_structure("EGFR")          # gene -> AlphaFold model
>>> ov.mol.view(s, color_by="pLDDT")            # interactive 3D, in-notebook
>>> ov.mol.plot_pae(s)                          # model-confidence map
>>> df = ov.mol.pockets(s)                      # binding pockets (rust-fpocket)
>>> ov.mol.druggability(s)                      # druggable / difficult verdict
>>> drugs = ov.mol.known_drugs("EGFR")          # ChEMBL known drugs
>>> val = ov.mol.redock_validate(ov.mol.fetch_structure("1M17", source="pdb"))
>>> result = ov.mol.dock(s, "gefitinib", pocket=1)
>>> ov.mol.view_docking(s, result)              # the binding pose, interactive

Surface
-------
Structure        ``fetch_structure`` (AlphaFold DB / RCSB PDB),
                 ``predict_structure`` (ESMFold API), ``MolStructure``
Visualization    ``view`` (py3Dmol interactive 3D — survives nbconvert),
                 ``view_docking``, ``plot_pae``
Druggability     ``pockets``, ``druggability`` (rust-fpocket backend)
Known drugs      ``known_drugs`` (ChEMBL)
Docking          ``dock``, ``redock_validate``, ``DockingResult``

The structural-biology stack (py3Dmol, biotite, chembl-webresource-client;
rdkit, vina, meeko for docking; fpocket-rs for pockets) is **optional** —
``import omicverse.mol`` does no heavy work, and each backend is gated by
an actionable ``ImportError`` (``pip install omicverse[mol]`` /
``omicverse[mol-dock]``). Follows the lazy-loading pattern of the rest of
omicverse.
"""
from __future__ import annotations

import importlib as _importlib

# Lazy public surface — single source of truth for what's exposed.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Structure acquisition
    "MolStructure":      ("._structure", "MolStructure"),
    "fetch_structure":   ("._structure", "fetch_structure"),
    "predict_structure": ("._structure", "predict_structure"),
    # Interactive visualization
    "view":              ("._view", "view"),
    "view_docking":      ("._view", "view_docking"),
    "plot_pae":          ("._view", "plot_pae"),
    # Pockets & druggability
    "pockets":           ("._pocket", "pockets"),
    "druggability":      ("._pocket", "druggability"),
    # Known drugs
    "known_drugs":       ("._drugs", "known_drugs"),
    # Docking
    "dock":              ("._dock", "dock"),
    "redock_validate":   ("._dock", "redock_validate"),
    "DockingResult":     ("._dock", "DockingResult"),
}

_REGISTRY_SUBMODULES = ("._structure", "._view", "._pocket", "._drugs",
                        "._dock")


def _hydrate_registry() -> None:
    """Force-import every ``@register_function``-bearing submodule so the
    global registry sees ``ov.mol`` at export time. Called from
    :func:`omicverse._registry._hydrate_registry_for_export`."""
    for mod in _REGISTRY_SUBMODULES:
        try:
            _importlib.import_module(mod, __name__)
        except Exception:
            # Optional backends may be missing — register what loads cleanly.
            continue


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(list(globals().keys()) + list(_LAZY_ATTRS.keys())))


__version__ = "0.1.0"

__all__ = list(_LAZY_ATTRS.keys())
