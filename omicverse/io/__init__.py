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
    - ``ov.io.read_visium(...)``, ``ov.io.read_visium_hd(...)``,
      ``ov.io.read_xenium(...)``, ``ov.io.read_stereoseq(...)`` — every reader
      in ``ov.io.spatial`` is reachable here under the same name
    - ``ov.io.read_fcs(...)``
    - ``ov.io.read_csv(...)``, ``ov.io.save(...)``, ``ov.io.load(...)``
"""

from . import bulk, cytometry, general, single, spatial
from .cytometry import parse_spillover, read_fcs
from .general import load, read_csv, read_table, save
from .single import read, read_10x_h5, read_10x_mtx, read_h5ad
# Every public spatial reader is re-exported here, deliberately. The list used
# to hold five of the eight, and the three it left out were `read_visium`,
# `read_atera` and `read_stereoseq` — so `ov.io.read_visium_hd` worked while
# `ov.io.read_visium`, the commonest call of the set, raised AttributeError.
# Nothing distinguished the five from the three; downstream code and docs read
# the namespace as uniform and wrote the shorter form, which then failed at
# runtime. Adding a reader to `spatial.__all__` without adding it here
# reintroduces that trap.
from .spatial import (
    bin_stereoseq,
    read_atera,
    read_nanostring,
    read_stereoseq,
    read_visium,
    read_visium_hd,
    read_visium_hd_bin,
    read_visium_hd_seg,
    read_xenium,
    write_visium_hd_cellseg,
)

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
    "read_visium",
    "read_visium_hd",
    "read_visium_hd_bin",
    "read_visium_hd_seg",
    "write_visium_hd_cellseg",
    "read_nanostring",
    "read_xenium",
    "read_atera",
    "read_stereoseq",
    "bin_stereoseq",
    "read_fcs",
    "parse_spillover",
    "read_csv",
    "read_table",
    "save",
    "load",
]
