"""Alias for the ``pylimma`` package (PyPI ``python-limma``; omicverse/py-limma).

omicverse used to vendor a copy of pylimma here. It now depends on the
released package; this module keeps ``omicverse.external.pylimma`` importable
for existing code and resolves to the installed ``pylimma``.
"""
import sys

import pylimma

sys.modules[__name__] = pylimma
