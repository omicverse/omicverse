r"""Cytometry I/O — Flow Cytometry Standard (``.fcs``) files.

A peer of ``io.bulk`` / ``io.single`` / ``io.spatial`` rather than a member of
one of them: an FCS event is a cell, but the file format, its metadata model
($PnN/$PnS, $SPILLOVER) and the instruments that write it are their own world,
shared by conventional flow, spectral flow and mass cytometry.
"""

from ._fcs import parse_spillover, read_fcs

__all__ = ["read_fcs", "parse_spillover"]
