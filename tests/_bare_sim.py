"""Simulate a base install — omicverse WITHOUT the optional extras.

The ``build`` CI job installs omicverse with no extras, so every dependency that
``omicverse[synbio]`` provides is absent there. A test that assumes one is present
does not fail locally, where the full environment is installed; it fails 26
minutes into CI, once per push.

Enable this plugin to get the same answer in about 40 seconds::

    pytest tests/ -k synbio -p tests._bare_sim

It hides the extras by intercepting ``__import__``, so the code under test takes
exactly the paths it takes on a base install: the graceful ``ImportError`` carrying
the ``pip install 'omicverse[synbio]'`` hint, or the ``pytest.importorskip`` that
skips the test. Anything that ends in a bare ``ModuleNotFoundError`` is a missing
gate.

**Not a substitute for CI.** It does not check that the package imports at all on a
base install, and it does not build a wheel. It checks the one thing that keeps
breaking: tests that quietly require an extra.

Written after this class of failure reached CI twice — once for the ``[synbio]``
extra in general, and again for ``cobra`` / ``pyreadr`` / ``dnachisel`` /
``python_codon_tables`` / ``esm`` across thirty new regression tests. The second
time, running under this plugin found eight more cases than CI had reported,
because CI stops at the first batch of errors.
"""
from __future__ import annotations

import builtins
import sys

#: Modules the optional extras provide. Hidden while this plugin is active.
HIDDEN = frozenset({
    "cobra", "optlang", "straindesign", "equilibrator_api",   # [synbio] layer A
    "dnachisel", "python_codon_tables", "primer3", "RNA",      # layer C / DNA
    "esm",                                                    # layer B
    "pyreadr",                                                # the R datasets
})

_real_import = builtins.__import__


def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in HIDDEN:
        raise ModuleNotFoundError(f"No module named {root!r}")
    return _real_import(name, globals, locals, fromlist, level)


def pytest_configure(config):
    builtins.__import__ = _blocked_import
    for module in list(sys.modules):
        if module.split(".")[0] in HIDDEN:
            del sys.modules[module]


def pytest_unconfigure(config):
    builtins.__import__ = _real_import
