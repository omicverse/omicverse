r"""
Input/Output utilities for OmicVerse datasets.

Subpackages:
    general: Shared I/O helpers for tabular data and serialization.
    bulk: I/O helpers for bulk omics resources.
    single: I/O helpers for single-cell data.
    spatial: I/O helpers for spatial omics data.
    cytometry: I/O helpers for flow / mass cytometry (.fcs).

Compatibility shortcuts:
    - ``ov.io.read(...)``
    - ``ov.io.read_h5ad(...)``
    - ``ov.io.read_10x_h5(...)``
    - ``ov.io.read_10x_mtx(...)``
    - ``ov.io.read_visium_hd(...)``
    - ``ov.io.read_fcs(...)``
    - ``ov.io.read_csv(...)``, ``ov.io.save(...)``, ``ov.io.load(...)``
"""

from . import bulk, cytometry, general, single, spatial
from .cytometry import parse_spillover, read_fcs
from .general import load, read_csv, read_table, save
from .single import read, read_10x_h5, read_10x_mtx, read_h5ad
from .spatial import read_nanostring, read_visium_hd, read_visium_hd_bin, read_visium_hd_seg, read_xenium

__all__ = [
    "general",
    "bulk",
    "single",
    "spatial",
    "cytometry",
    # top-level compatibility exports
    "read",
    "read_h5ad",
    "read_10x_h5",
    "read_10x_mtx",
    "read_visium_hd",
    "read_visium_hd_bin",
    "read_visium_hd_seg",
    "read_nanostring",
    "read_xenium",
    "read_fcs",
    "parse_spillover",
    "read_csv",
    "read_table",
    "save",
    "load",
]
