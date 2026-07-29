"""Every public reader in an ``omicverse.io`` submodule is reachable on ``ov.io``.

``ov.io`` re-exports its submodules' readers as top-level shortcuts, and the
list was maintained by hand. It drifted: ``read_visium_hd``, ``read_xenium``
and ``read_nanostring`` were re-exported while ``read_visium``, ``read_atera``
and ``read_stereoseq`` were not, so ``ov.io.read_visium`` — the commonest call
of the set — raised ``AttributeError`` while its Visium-HD sibling worked.

Nothing in the namespace signalled which names were shortcuts and which were
not, so downstream code and documentation read it as uniform and wrote the
short form. The failure is invisible until the line actually runs, and the
error names an attribute rather than the real problem, so it reads like a
version or install issue.

A hand-maintained mirror needs a test that notices when the mirror stops
matching. This walks the submodules' own ``__all__`` rather than a second
hand-written list, so adding a reader to a submodule is enough to make this
test demand the shortcut too.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

import omicverse as ov

# Submodules whose public surface is mirrored on ``ov.io``. ``bulk`` and
# ``general`` are excluded deliberately: ``general`` holds the format-agnostic
# helpers that are already listed by hand, and ``bulk`` has no reader worth a
# shortcut. Add a submodule here when its readers start being referenced as
# ``ov.io.<name>``.
MIRRORED_SUBMODULES = ("spatial", "cytometry", "single")

_IO_DIR = pathlib.Path(ov.io.__file__).parent


def _declared_all(submodule: str) -> list[str]:
    """Read ``__all__`` from source, without importing the submodule.

    Reading the AST keeps this test runnable when a submodule's heavy optional
    backend is absent — the point is the export list, not the implementations.
    """
    tree = ast.parse((_IO_DIR / submodule / "__init__.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return list(ast.literal_eval(node.value))
    raise AssertionError(f"omicverse/io/{submodule}/__init__.py declares no __all__")


def _is_reader(name: str) -> bool:
    """Only readers are mirrored on the facade, and only readers are the bug.

    A submodule also exports conversion helpers (``convert_to_pandas``,
    ``wrap_dataframe``, ...). Those are deliberately *not* shortcuts: nobody
    writes ``ov.io.convert_to_pandas``, and widening the facade to cover them
    is an API decision, not a drift repair. The drift that actually bites is a
    reader that is public in its submodule but absent from the facade, because
    the reader names are what documentation and downstream code reach for.
    """
    return name.startswith(("read_", "bin_", "write_"))


def _mirrored_names() -> list[tuple[str, str]]:
    pairs = []
    for submodule in MIRRORED_SUBMODULES:
        for name in _declared_all(submodule):
            if _is_reader(name):
                pairs.append((submodule, name))
    return pairs


def test_the_audit_has_something_to_audit():
    """Guard against the AST reader silently returning nothing.

    Without this, a parse change would make every case below vacuous and the
    suite would go green while checking nothing.
    """
    pairs = _mirrored_names()
    assert len(pairs) > 10, f"expected a real export surface, found {pairs}"
    assert ("spatial", "read_visium") in pairs


@pytest.mark.parametrize("submodule,name", _mirrored_names(),
                         ids=lambda v: v if isinstance(v, str) else str(v))
def test_submodule_export_is_reachable_on_the_facade(submodule: str, name: str):
    assert hasattr(ov.io, name), (
        f"ov.io.{name} does not resolve, but omicverse/io/{submodule} exports it. "
        f"Add it to the imports and __all__ in omicverse/io/__init__.py — a name "
        f"that is public in a submodule but missing from the facade fails only "
        f"when the line runs, with an AttributeError that hides the cause."
    )


@pytest.mark.parametrize("submodule,name", _mirrored_names(),
                         ids=lambda v: v if isinstance(v, str) else str(v))
def test_facade_name_is_the_submodule_object(submodule: str, name: str):
    """The shortcut must BE the submodule's object, not a same-named other one."""
    if not hasattr(ov.io, name):
        pytest.skip("reachability is covered by the test above")
    facade = getattr(ov.io, name)
    original = getattr(getattr(ov.io, submodule), name)
    assert facade is original, (
        f"ov.io.{name} is not ov.io.{submodule}.{name} — the shortcut points "
        f"somewhere else, so the two call sites can diverge silently."
    )


def test_every_facade_export_resolves():
    """The other direction: nothing in ``ov.io.__all__`` is a dead name."""
    missing = [n for n in ov.io.__all__ if not hasattr(ov.io, n)]
    assert not missing, f"ov.io.__all__ lists names that do not resolve: {missing}"
