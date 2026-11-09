"""Alias for the ``pyedger`` package (PyPI; omicverse/py-edgeR).

omicverse used to vendor a copy of pyedger here. It now depends on the
released package; this module keeps ``omicverse.external.pyedger`` importable
for existing code and resolves to the installed ``pyedger``.
"""
import sys

import pyedger

sys.modules[__name__] = pyedger
