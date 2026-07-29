"""Everything ``ov.synbio`` advertises must actually resolve.

``omicverse/synbio/__init__.py`` is a lazy facade: ``__getattr__`` looks a name up
in ``_LAZY_ATTRS`` and raises ``AttributeError`` for anything absent from it.
That map is maintained by hand and each submodule's ``__all__`` is maintained
separately, so the two drift — and the drift is invisible, because nothing
imports the facade and the submodule together. Thirty-three names shipped in
2.3.0 that were listed in their module's ``__all__``, documented, and simply not
reachable as ``ov.synbio.<name>``.

The check is deliberately AST-only on the ``__all__`` side: the ``[synbio]``
extra (cobra, fair-esm, ViennaRNA…) is absent from the ``build`` CI job, and an
export-map audit that skipped there would be worthless. It does not need those
packages either, because every heavy import in ``ov.synbio`` lives inside a
function body rather than at module top level — which the last test here pins,
since a single top-level ``import cobra`` would quietly turn the whole audit into
a skip.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

import omicverse.synbio as sb

SYNBIO_DIR = pathlib.Path(sb.__file__).parent


def _literal_all(path: pathlib.Path):
    """A module's ``__all__`` read from source, without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                return list(ast.literal_eval(value))
    return None


def _submodules():
    return sorted(p for p in SYNBIO_DIR.glob("*.py") if p.name != "__init__.py")


def _exported():
    """(module stem, name) for every name in every submodule's ``__all__``."""
    pairs = []
    for path in _submodules():
        for name in _literal_all(path) or ():
            pairs.append((path.stem, name))
    return pairs


EXPORTED = _exported()


def test_the_audit_actually_found_something_to_audit():
    """A guard on the guard: an AST change that silently returns nothing here
    would make every parametrised case below vacuous and the suite still green."""
    assert len(EXPORTED) > 250, f"only {len(EXPORTED)} exported names found"
    assert len({m for m, _ in EXPORTED}) > 40


@pytest.mark.parametrize("module,name", EXPORTED, ids=lambda v: str(v))
def test_every_submodule_export_is_reachable_from_the_facade(module, name):
    """``<module>.__all__`` promises it, so ``ov.synbio.<name>`` must deliver it.

    Failure means the name is missing from ``_LAZY_ATTRS``: the docs and the
    module both advertise it and the facade raises ``AttributeError``.
    """
    assert name in sb._LAZY_ATTRS, (
        f"{module}.py exports {name!r} in __all__ but it is missing from "
        f"_LAZY_ATTRS, so ov.synbio.{name} raises AttributeError")
    assert getattr(sb, name) is not None


def test_facade_all_matches_the_lazy_map():
    """``__all__`` is built from the map, so ``dir()`` and ``__all__`` agree."""
    assert set(sb.__all__) == set(sb._LAZY_ATTRS)
    assert set(sb.__all__) <= set(dir(sb))


@pytest.mark.parametrize("name", sorted(sb._LAZY_ATTRS))
def test_every_lazy_map_entry_resolves(name):
    """The other direction: a map entry pointing at a moved or renamed
    attribute is just as broken, and equally invisible until someone asks."""
    module_path, attr = sb._LAZY_ATTRS[name]
    assert getattr(sb, name) is not None, f"{name} -> {module_path}.{attr}"


def test_no_submodule_imports_a_heavy_dependency_at_module_level():
    """The lazy facade only buys anything if the submodules are lazy too.

    ``import omicverse.synbio`` must stay free of cobra/torch/ViennaRNA, both so
    that ``import omicverse`` is cheap and so that the reachability audit above
    runs on an install without the ``[synbio]`` extra rather than erroring.
    """
    heavy = {"cobra", "esm", "torch", "RNA", "primer3", "dnachisel", "Bio",
             "sbol3", "pyreadr", "micom", "equilibrator_api", "py3Dmol"}
    offenders = []
    for path in _submodules():
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                offenders += [(path.name, a.name) for a in node.names
                              if a.name.split(".")[0] in heavy]
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in heavy:
                    offenders.append((path.name, node.module))
    assert not offenders, f"top-level heavy imports: {offenders}"
