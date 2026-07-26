r"""Typed result objects for :mod:`omicverse.mol.md`.

:class:`MDSystem` is a simulatable system (topology + forcefield-built
``openmm.System`` + coordinates); :class:`MDTrajectory` is a handle on a
produced trajectory plus the metadata needed to reproduce it.

Both carry a ``provenance`` dict — engine versions, forcefield, integrator,
thermostat/barostat settings, seed, platform, and input/output file hashes —
so a run can be traced and repeated.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ._common import check_mdtraj, file_hash


class MDSystem:
    """A prepared, simulatable molecular system.

    Attributes
    ----------
    topology : openmm.app.Topology
        Atoms, residues, bonds and (for explicit solvent) the periodic box.
    system : openmm.System
        The parameterised system (forces, constraints, particle masses).
    positions : openmm.unit.Quantity
        Coordinates of every particle.
    box_vectors : openmm.unit.Quantity or None
        Periodic box; ``None`` for implicit solvent / vacuum.
    forcefield : str
        Name of the protein forcefield used (e.g. ``'amber14-all'``).
    water : str
        Water model, or ``'implicit'``.
    provenance : dict
        Full record of how the system was built (see module docstring).
    """

    def __init__(self, topology, system, positions, *, box_vectors=None,
                 forcefield: str = "amber14-all", water: str = "tip3p",
                 provenance: Optional[Dict[str, Any]] = None):
        self.topology = topology
        self.system = system
        self.positions = positions
        self.box_vectors = box_vectors
        self.forcefield = forcefield
        self.water = water
        self.provenance: Dict[str, Any] = provenance or {}

    @property
    def n_atoms(self) -> int:
        """Number of particles in the system (including solvent)."""
        return int(self.system.getNumParticles())

    @property
    def is_periodic(self) -> bool:
        """Whether the system uses a periodic box (explicit solvent)."""
        return self.box_vectors is not None

    def save(self, prefix: str) -> Dict[str, str]:
        """Write the system to ``<prefix>.pdb`` / ``.system.xml`` / ``.state.xml``.

        The PDB holds topology + current coordinates; the XML files serialise
        the parameterised ``openmm.System`` and a ``State`` (positions + box),
        which together let a run be restarted without re-running
        :func:`~omicverse.mol.prepare_system`.

        Returns a dict of the written paths.
        """
        import openmm
        from openmm import app

        d = os.path.dirname(os.path.abspath(prefix))
        if d:
            os.makedirs(d, exist_ok=True)

        pdb_path = f"{prefix}.pdb"
        with open(pdb_path, "w") as fh:
            app.PDBFile.writeFile(self.topology, self.positions, fh,
                                  keepIds=True)

        sys_path = f"{prefix}.system.xml"
        with open(sys_path, "w") as fh:
            fh.write(openmm.XmlSerializer.serialize(self.system))

        # A minimal context just to snapshot positions/box as a State.
        integrator = openmm.VerletIntegrator(0.001)
        context = openmm.Context(self.system, integrator,
                                 openmm.Platform.getPlatformByName("Reference"))
        context.setPositions(self.positions)
        if self.box_vectors is not None:
            context.setPeriodicBoxVectors(*self.box_vectors)
        state = context.getState(getPositions=True, getVelocities=True)
        state_path = f"{prefix}.state.xml"
        with open(state_path, "w") as fh:
            fh.write(openmm.XmlSerializer.serialize(state))
        del context, integrator

        return {"pdb": pdb_path, "system": sys_path, "state": state_path}

    def __repr__(self) -> str:
        solvent = self.water if self.water != "implicit" else "implicit"
        box = ""
        if self.box_vectors is not None:
            try:
                from openmm import unit
                v = self.box_vectors.value_in_unit(unit.nanometer)
                box = f", box {v[0][0]:.1f}x{v[1][1]:.1f}x{v[2][2]:.1f} nm"
            except Exception:
                box = ", periodic"
        return (f"MDSystem({self.n_atoms} atoms, {self.forcefield}/{solvent}"
                f"{box})")


class MDTrajectory:
    """A produced trajectory on disk plus its metadata.

    Attributes
    ----------
    traj_path : str
        Trajectory file (``.dcd`` or ``.xtc``).
    top_path : str
        Topology PDB written alongside the trajectory.
    n_frames : int
        Frames written.
    dt_ps : float
        Time between saved frames, in picoseconds.
    provenance : dict
        Integrator/ensemble/platform/seed record plus file hashes.
    """

    def __init__(self, traj_path: str, top_path: str, *, n_frames: int = 0,
                 dt_ps: float = 0.0, checkpoint: Optional[str] = None,
                 log_path: Optional[str] = None,
                 provenance: Optional[Dict[str, Any]] = None):
        self.traj_path = traj_path
        self.top_path = top_path
        self.n_frames = int(n_frames)
        self.dt_ps = float(dt_ps)
        self.checkpoint = checkpoint
        self.log_path = log_path
        self.provenance: Dict[str, Any] = provenance or {}

    @property
    def time_ns(self) -> float:
        """Total simulated time represented by the saved frames, in ns."""
        return self.n_frames * self.dt_ps / 1000.0

    def load(self, stride: int = 1, *, top: Optional[str] = None):
        """Load the trajectory with MDTraj.

        Parameters
        ----------
        stride
            Read every ``stride``-th frame (memory control for long runs).
        top
            Override the topology file.

        Notes
        -----
        Coordinates are **not** unwrapped here. For any distance-based
        analysis on a periodic system call :meth:`wrap` first (or use the
        ``ov.mol`` analysis helpers, which wrap on demand).
        """
        check_mdtraj()
        import mdtraj as md
        return md.load(self.traj_path, top=top or self.top_path, stride=stride)

    def wrap(self, out_path: Optional[str] = None,
             *, anchor_molecules: bool = True) -> "MDTrajectory":
        """Image molecules back into the primary unit cell (PBC fix).

        Without this, a molecule that diffuses across a periodic boundary
        appears to jump, which corrupts RMSD, contacts and hydrogen-bond
        analysis. Returns a **new** :class:`MDTrajectory` pointing at the
        wrapped file; the original is untouched.
        """
        check_mdtraj()
        import mdtraj as md

        traj = md.load(self.traj_path, top=self.top_path)
        if traj.unitcell_vectors is None:
            # Nothing to do for a non-periodic (implicit-solvent) run.
            return self
        anchors = None
        if anchor_molecules:
            try:
                anchors = [set(traj.top.residue(0).atoms)]
                # anchor on the protein as one molecule when we can find it
                protein = traj.top.select("protein")
                if len(protein):
                    anchors = [set(traj.top.atom(i) for i in protein)]
            except Exception:
                anchors = None
        try:
            traj = traj.image_molecules(anchor_molecules=anchors)
        except Exception:
            traj = traj.image_molecules()

        if out_path is None:
            stem, ext = os.path.splitext(self.traj_path)
            out_path = f"{stem}_wrapped{ext or '.dcd'}"
        traj.save(out_path)

        prov = dict(self.provenance)
        prov["wrapped"] = True
        prov["wrapped_from"] = self.traj_path
        out = MDTrajectory(out_path, self.top_path, n_frames=traj.n_frames,
                           dt_ps=self.dt_ps, checkpoint=self.checkpoint,
                           log_path=self.log_path, provenance=prov)
        out.provenance["files"] = {
            "trajectory": out_path, "trajectory_sha256": file_hash(out_path),
            "topology": self.top_path,
        }
        return out

    def __repr__(self) -> str:
        return (f"MDTrajectory({self.n_frames} frames, "
                f"{self.time_ns:.3f} ns @ {self.dt_ps:g} ps/frame, "
                f"'{os.path.basename(self.traj_path)}')")
