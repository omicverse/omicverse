r"""Molecular dynamics for ``ov.mol`` — OpenMM engine, MDTraj analysis.

This subpackage gives the otherwise-static ``ov.mol`` stack a time dimension:
a structure (fetched, predicted, or docked) becomes a solvated system, is
minimised and equilibrated, runs production dynamics on the GPU, and is then
analysed — up to a dynamics-averaged binding free energy.

Everything here is re-exported on ``ov.mol`` (``ov.mol.simulate``,
``ov.mol.rmsd``, ``ov.mol.mmgbsa``, ...); import it from there.

Layout
------
``_common``    dependency gates, OpenMM platform/device selection, provenance
``_types``     :class:`MDSystem`, :class:`MDTrajectory`
``_system``    :func:`prepare_system` (repair, parameterise, solvate)
``_run``       :func:`minimize`, :func:`equilibrate`, :func:`production`,
               :func:`simulate`
``_analysis``  RMSD/RMSF/Rg/H-bonds/contacts/DSSP/clustering/PCA + plots
``_mmgbsa``    :func:`mmgbsa`, :class:`MMGBSAResult`

No heavy dependency is imported at module scope — ``openmm``, ``mdtraj`` and
the ligand-parameterisation stack are pulled in inside the functions that use
them, so importing ``omicverse.mol`` costs nothing and works without the
``[md]`` extra installed.
"""

from __future__ import annotations

from ._analysis import (cluster, contacts, hbonds, load_trajectory,
                        native_contacts, pca, plot_energy, plot_rg, plot_rmsd,
                        plot_rmsf, radius_of_gyration, rmsd, rmsd_matrix, rmsf,
                        secondary_structure)
from ._mmgbsa import MMGBSAResult, alchemical_free_energy, mmgbsa
from ._run import equilibrate, minimize, production, simulate
from ._system import prepare_system
from ._types import MDSystem, MDTrajectory

__all__ = [
    # types
    "MDSystem", "MDTrajectory", "MMGBSAResult",
    # preparation + dynamics
    "prepare_system", "minimize", "equilibrate", "production", "simulate",
    # analysis
    "load_trajectory", "rmsd", "rmsf", "radius_of_gyration", "hbonds",
    "contacts", "native_contacts", "secondary_structure", "rmsd_matrix",
    "cluster", "pca",
    # plotting
    "plot_rmsd", "plot_rmsf", "plot_rg", "plot_energy",
    # free energy
    "mmgbsa", "alchemical_free_energy",
]
