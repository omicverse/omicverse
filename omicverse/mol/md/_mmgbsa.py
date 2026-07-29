r"""End-point binding free energy (MM-GBSA) for :mod:`omicverse.mol.md`.

This is the bridge that turns a *static* docking score into a
*dynamics-averaged* binding estimate: take an MD trajectory of the complex,
and for each frame evaluate

.. math::

   \Delta G_\mathrm{bind} = \langle G_\mathrm{complex}\rangle
                          - \langle G_\mathrm{receptor}\rangle
                          - \langle G_\mathrm{ligand}\rangle

with :math:`G = E_\mathrm{vdW} + E_\mathrm{elec} + G_\mathrm{polar} +
G_\mathrm{nonpolar}`. The single-trajectory approximation is used (receptor
and ligand conformations are taken from the complex), so bonded terms cancel
exactly.

.. warning::
   End-point MM-GBSA is a **relative ranking** tool. It systematically
   overestimates affinities, ignores explicit-water and entropy contributions,
   and must not be read as an experimental binding free energy. For rigorous
   free energies use alchemical FEP/TI (see :func:`alchemical_free_energy`).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import numpy as np

from ..._registry import register_function
from ._common import base_provenance, check_md, check_mdtraj
from ._types import MDTrajectory

# AMBER's standard non-polar surface tension: 0.0072 kcal/(mol.A^2)
_GAMMA_KJ_PER_NM2 = 0.0072 * 4.184 * 100.0     # -> kJ/(mol.nm^2)
_KJ_TO_KCAL = 1.0 / 4.184

_SOLVENT_RESIDUES = {"HOH", "WAT", "TIP3", "TIP4", "SOL", "NA", "CL", "K",
                     "MG", "CA", "ZN", "SOD", "CLA", "POT", "ION"}


class MMGBSAResult:
    """Output of :func:`mmgbsa` — a dynamics-averaged binding free energy.

    Attributes
    ----------
    dG : float
        Mean binding free energy, kcal/mol (negative = favourable).
    std : float
        Standard deviation across frames, kcal/mol.
    sem : float
        Standard error of the mean, kcal/mol.
    per_frame : numpy.ndarray
        Per-frame ``dG``, kcal/mol.
    components : dict
        Mean of each term (``vdw``, ``electrostatic``, ``polar_solvation``,
        ``nonpolar_solvation``), kcal/mol.
    n_frames : int
        Frames used.
    provenance : dict
        Method, forcefield, frame selection, and the source trajectory.
    """

    def __init__(self, per_frame, components, *, n_frames: int,
                 provenance: Optional[Dict[str, Any]] = None):
        self.per_frame = np.asarray(per_frame, dtype=float)
        self.components = components
        self.n_frames = int(n_frames)
        self.provenance = provenance or {}

    @property
    def dG(self) -> float:
        return float(np.mean(self.per_frame)) if len(self.per_frame) else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.per_frame, ddof=1)) if len(self.per_frame) > 1 else 0.0

    @property
    def sem(self) -> float:
        n = len(self.per_frame)
        return float(self.std / np.sqrt(n)) if n > 1 else 0.0

    def to_frame(self):
        """Component breakdown as a one-row :class:`pandas.DataFrame`."""
        import pandas as pd
        row = {"dG_kcal_mol": self.dG, "std": self.std, "sem": self.sem,
               "n_frames": self.n_frames}
        row.update({f"{k}_kcal_mol": v for k, v in self.components.items()})
        return pd.DataFrame([row])

    def __repr__(self) -> str:
        return (f"MMGBSAResult(dG = {self.dG:.2f} +/- {self.sem:.2f} kcal/mol, "
                f"{self.n_frames} frames)")


def _split_indices(topology):
    """Indices of receptor (protein) and ligand atoms in an OpenMM topology."""
    receptor, ligand, solvent = [], [], []
    for atom in topology.atoms():
        rname = atom.residue.name.upper()
        if rname in _SOLVENT_RESIDUES:
            solvent.append(atom.index)
        elif atom.residue.chain is not None and _is_polymer(atom.residue):
            receptor.append(atom.index)
        else:
            ligand.append(atom.index)
    return np.array(receptor), np.array(ligand), np.array(solvent)


_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
       "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN", "ACE", "NME"}


def _is_polymer(residue) -> bool:
    return residue.name.upper() in _AA


def _zero_charges(system):
    """Copy of ``system`` with all partial charges zeroed (isolates vdW)."""
    import copy

    import openmm
    out = copy.deepcopy(system)
    for force in out.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for i in range(force.getNumParticles()):
                q, sigma, eps = force.getParticleParameters(i)
                force.setParticleParameters(i, 0.0, sigma, eps)
            for i in range(force.getNumExceptions()):
                p1, p2, qq, sigma, eps = force.getExceptionParameters(i)
                force.setExceptionParameters(i, p1, p2, 0.0, sigma, eps)
    return out


def _tag_force_groups(system) -> None:
    """Group 1 = non-bonded (vdW+elec), group 2 = GB solvation, 0 = the rest."""
    import openmm
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setForceGroup(1)
        elif isinstance(force, (openmm.GBSAOBCForce, openmm.CustomGBForce)):
            force.setForceGroup(2)
        else:
            force.setForceGroup(0)


class _EnergyEvaluator:
    """Evaluates MM + GB energies of one molecular species, frame by frame."""

    def __init__(self, topology, system, platform, props):
        import openmm
        from openmm import unit

        _tag_force_groups(system)
        self.topology = topology
        self._unit = unit
        self._ctx = openmm.Context(
            system, openmm.VerletIntegrator(0.001 * unit.picoseconds),
            platform, props)
        # a charge-free twin isolates the van der Waals part
        vdw_system = _zero_charges(system)
        _tag_force_groups(vdw_system)
        self._ctx_vdw = openmm.Context(
            vdw_system, openmm.VerletIntegrator(0.001 * unit.picoseconds),
            platform, props)

    def energies(self, positions_nm) -> Dict[str, float]:
        """``{'vdw', 'electrostatic', 'polar_solvation'}`` in kJ/mol."""
        unit = self._unit
        pos = positions_nm * unit.nanometer

        self._ctx.setPositions(pos)
        nonbonded = self._ctx.getState(getEnergy=True, groups={1}) \
            .getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        polar = self._ctx.getState(getEnergy=True, groups={2}) \
            .getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        self._ctx_vdw.setPositions(pos)
        vdw = self._ctx_vdw.getState(getEnergy=True, groups={1}) \
            .getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        return {"vdw": vdw, "electrostatic": nonbonded - vdw,
                "polar_solvation": polar}


def _sasa_nm2(traj_slice) -> np.ndarray:
    """Per-frame solvent-accessible surface area (nm^2) via Shrake-Rupley."""
    import mdtraj as md
    return md.shrake_rupley(traj_slice, mode="atom").sum(axis=1)


def _build_species_systems(top_pdb: str, ligand_smiles: Optional[str],
                           forcefield_name: str, ligand_ff: str):
    """Build implicit-solvent Systems for complex / receptor / ligand."""
    from openmm import app

    from ._system import (_FF_FILES, _IMPLICIT_XML, _norm_ff,
                          _register_ligand_template, _to_openff_molecule)

    pdb = app.PDBFile(top_pdb)
    ff_name = _norm_ff(forcefield_name)
    forcefield = app.ForceField(_FF_FILES[ff_name], _IMPLICIT_XML)

    if ligand_smiles:
        off_mol = _to_openff_molecule(ligand_smiles)
        _register_ligand_template(forcefield, off_mol, ligand_ff)

    receptor_idx, ligand_idx, solvent_idx = _split_indices(pdb.topology)
    if len(ligand_idx) == 0:
        raise ValueError(
            "no ligand found in the trajectory topology — MM-GBSA needs a "
            "protein-ligand complex. Simulate the complex (e.g. "
            "ov.mol.simulate(docking_result, ...)) before scoring.")

    def _subset(keep_idx):
        """Modeller-built topology/system containing only ``keep_idx``."""
        modeller = app.Modeller(pdb.topology, pdb.positions)
        drop = [a for a in modeller.topology.atoms()
                if a.index not in set(int(i) for i in keep_idx)]
        modeller.delete(drop)
        system = forcefield.createSystem(
            modeller.topology, nonbondedMethod=app.NoCutoff,
            constraints=None, rigidWater=False)
        return modeller.topology, system

    solute_idx = np.concatenate([receptor_idx, ligand_idx])
    complex_top, complex_sys = _subset(solute_idx)
    receptor_top, receptor_sys = _subset(receptor_idx)
    ligand_top, ligand_sys = _subset(ligand_idx)

    return {
        "complex": (complex_top, complex_sys, solute_idx),
        "receptor": (receptor_top, receptor_sys, receptor_idx),
        "ligand": (ligand_top, ligand_sys, ligand_idx),
    }


def _frame_selection(n_frames: int, frames: Union[int, str]) -> np.ndarray:
    if isinstance(frames, str):
        if frames.lower() != "all":
            raise ValueError(f"frames must be an int or 'all', got {frames!r}")
        return np.arange(n_frames)
    k = int(frames)
    if k <= 0:
        raise ValueError("frames must be positive")
    if k >= n_frames:
        return np.arange(n_frames)
    return np.unique(np.linspace(0, n_frames - 1, k).astype(int))


@register_function(
    aliases=["mmgbsa", "MM-GBSA", "结合自由能", "binding_free_energy",
             "mmpbsa", "rescore", "结合能"],
    category="mol",
    description=(
        "Dynamics-averaged binding free energy by end-point MM-GBSA. Takes "
        "an MD trajectory of a protein-ligand complex (or a DockingResult, "
        "which it can relax with a short MD first) and computes "
        "dG = G_complex - G_receptor - G_ligand over sampled frames with "
        "GBn2 implicit solvent, returning the mean, its spread, and the "
        "van der Waals / electrostatic / polar / non-polar breakdown. This "
        "is the dock -> MD -> rescore upgrade: it replaces a single static "
        "docking score with an ensemble estimate. Treat the number as a "
        "RELATIVE ranking, not an experimental affinity."
    ),
    examples=[
        "res = ov.mol.mmgbsa(traj)",
        "res = ov.mol.mmgbsa(docking_result, md_refine_ns=0.5)",
        "res.to_frame()",
    ],
    related=["mol.simulate", "mol.dock", "mol.production", "mol.contacts"],
)
def mmgbsa(
    source,
    *,
    receptor=None,
    ligand=None,
    method: str = "gbsa",
    frames: Union[int, str] = 50,
    md_refine_ns: float = 0.0,
    forcefield: str = "amber14-all",
    ligand_ff: str = "gaff-2.11",
    platform: str = "auto",
    precision: str = "mixed",
    outdir: str = "mmgbsa_out",
    seed: int = 0,
    verbose: bool = True,
) -> MMGBSAResult:
    r"""Estimate a binding free energy from a complex trajectory.

    Parameters
    ----------
    source
        An :class:`~omicverse.mol.md.MDTrajectory` of the **complex**, or a
        :class:`~omicverse.mol.DockingResult` (prepared and, if
        ``md_refine_ns > 0``, relaxed with a short MD first).
    receptor, ligand
        Override the receptor / ligand when they cannot be inferred.
        ``receptor`` is only consulted on the ``DockingResult`` path (an
        ``MDTrajectory`` already contains the complex, so it is ignored
        there); ``ligand`` is only used to recover the ligand SMILES when the
        trajectory's provenance does not carry one.
    method
        ``'gbsa'`` (GBn2 implicit solvent) — the only implemented method.
        ``'pbsa'`` / ``'mmpbsa'`` **raise** :class:`NotImplementedError`: they
        need an external Poisson-Boltzmann solver (APBS, AMBER's MMPBSA.py)
        that omicverse does not bundle, and there is no silent fallback to
        GBSA. Note that ``'mmpbsa'`` is also a registry *alias* of this
        function, so ``ov.find_function('mmpbsa')`` points here — at GBSA.
    frames
        Number of evenly spaced frames to score, or ``'all'``.
    md_refine_ns
        For a ``DockingResult``: length of the relaxation MD run between
        minimisation and scoring. ``0`` does **not** mean "no dynamics" —
        MM-GBSA averages over a *trajectory*, and there is no path here that
        scores an ``MDSystem``'s coordinates directly, so the floor is the
        shortest run that produces frames at all: minimise, then 2 ps of NVT,
        out of which ``frames`` snapshots are scored. Ignored entirely when
        ``source`` is already an
        :class:`~omicverse.mol.md.MDTrajectory` (that trajectory is scored
        as given).

    Returns
    -------
    MMGBSAResult

    Notes
    -----
    Single-trajectory approximation: receptor and ligand are taken from the
    complex, so internal bonded energies cancel. No entropy term is included.
    """
    check_md()
    check_mdtraj()
    import mdtraj as md

    if str(method).lower() in ("pbsa", "mmpbsa"):
        raise NotImplementedError(
            "MM-PBSA needs an external Poisson-Boltzmann solver (APBS / "
            "AMBER's MMPBSA.py), which omicverse does not bundle. Use "
            "method='gbsa' (GBn2), which is the fast standard for relative "
            "ranking.")
    if str(method).lower() != "gbsa":
        raise ValueError(f"method must be 'gbsa', got {method!r}")

    from ._common import resolve_platform

    # ---- get a complex trajectory --------------------------------------
    traj_obj: MDTrajectory
    if isinstance(source, MDTrajectory):
        traj_obj = source
    elif hasattr(source, "poses") and hasattr(source, "affinities"):
        from ._run import minimize, production
        from ._system import prepare_system

        os.makedirs(outdir, exist_ok=True)
        rec = receptor if receptor is not None else getattr(source, "receptor", None)
        if rec is None:
            # DockingResult.receptor arrived with the MD layer in 2.3.0, so a
            # result pickled by an earlier omicverse has no such attribute and
            # lands here. Re-docking is not required — the receptor is the only
            # missing piece, and passing it costs nothing.
            raise ValueError(
                "this DockingResult carries no receptor, so the complex "
                "cannot be rebuilt for scoring. Pass it explicitly:\n"
                "    ov.mol.mmgbsa(result, receptor=<the MolStructure you "
                "docked into>)\n"
                "Results from ov.mol.dock() on omicverse >= 2.3.0 carry the "
                "receptor themselves; one saved by an older version does not, "
                "so either supply receptor= or re-run ov.mol.dock().")
        if verbose:
            print("   preparing the docked complex (implicit solvent)")
        mdsys = prepare_system(rec, ligand=source, water="implicit",
                               ligand_ff=ligand_ff, forcefield=forcefield,
                               seed=seed, verbose=verbose)
        mdsys = minimize(mdsys, platform=platform, precision=precision,
                         verbose=verbose)
        run_ns = max(float(md_refine_ns), 0.0)
        if run_ns <= 0:
            # md_refine_ns=0 is "don't refine", but scoring needs a trajectory
            # on disk (see the md.load below), so the floor is the shortest
            # production that writes any frames: 2 ps of NVT off the minimised
            # pose. NB `report` is report_ps — picoseconds — so 0.002 here is
            # one 2 fs step, i.e. a frame every step (~1000 frames from those
            # 2 ps), and `frames` then picks how many are actually scored. A
            # single end-of-run frame would need report=2.0; that is left
            # alone deliberately, and not only for compatibility. Scoring one
            # frame returns std=0 and sem=0 — a spread of exactly zero, which
            # reads as precision this method does not have. MM-GBSA is only
            # good for relative ranking, so the ensemble that gives `sem` a
            # meaning is worth more than making the frame count match the
            # word "single". Changing it would also move every dG this path
            # has produced.
            run_ns, report = 0.002, 0.002
        else:
            report = max(run_ns * 1000.0 / max(int(frames if isinstance(frames, int) else 50), 1), 0.1)
        traj_obj = production(
            mdsys, ns=run_ns, ensemble="NVT", report_ps=report,
            traj=os.path.join(outdir, "complex.dcd"), platform=platform,
            precision=precision, seed=seed, progress=False, verbose=verbose)
    else:
        raise TypeError(
            "source must be an MDTrajectory of the complex or a "
            f"DockingResult — got {type(source).__name__}")

    # ---- systems for the three species ---------------------------------
    lig_smiles = None
    prov_sys = traj_obj.provenance.get("system_provenance", {})
    lig_smiles = prov_sys.get("ligand_smiles")
    if lig_smiles is None and ligand is not None:
        from ._system import _to_openff_molecule
        lig_smiles = _to_openff_molecule(ligand).to_smiles()
    ff_used = prov_sys.get("forcefield", forcefield)

    species = _build_species_systems(traj_obj.top_path, lig_smiles, ff_used,
                                     ligand_ff)

    # Energy evaluation builds its own Contexts, so it needs the same
    # device-failure fallback as the dynamics engine — a platform can load
    # fine and still fail when it compiles kernels (e.g. a CUDA/driver
    # version mismatch).
    from ._common import explicit_device_requested, is_device_error

    def _build(plat, props):
        return {name: _EnergyEvaluator(top, sysobj, plat, props)
                for name, (top, sysobj, _) in species.items()}

    plat, props = resolve_platform(platform, precision, verbose=verbose)
    try:
        evaluators = _build(plat, props)
    except Exception as exc:
        if explicit_device_requested(platform) or not is_device_error(exc):
            raise
        import warnings
        warnings.warn(
            f"{plat.getName()} platform failed while setting up MM-GBSA "
            f"energy evaluation ({exc}); falling back to a slower device.",
            RuntimeWarning, stacklevel=2)
        evaluators = None
        for fallback in ("OpenCL", "CPU"):
            if fallback == plat.getName():
                continue
            try:
                plat, props = resolve_platform(fallback, precision,
                                               verbose=verbose)
                evaluators = _build(plat, props)
                break
            except Exception:
                continue
        if evaluators is None:
            raise RuntimeError(
                "no OpenMM platform could evaluate MM-GBSA energies") from exc

    # ---- score the frames ----------------------------------------------
    traj = md.load(traj_obj.traj_path, top=traj_obj.top_path)
    if traj.unitcell_vectors is not None:
        from ._analysis import _image
        traj = _image(traj)
    picks = _frame_selection(traj.n_frames, frames)
    if verbose:
        print(f"   scoring {len(picks)} of {traj.n_frames} frames "
              f"(GBn2 implicit solvent)")

    terms = {k: [] for k in ("vdw", "electrostatic", "polar_solvation",
                             "nonpolar_solvation")}
    per_frame = []

    sasa_cache = {}
    for name, (_, _, idx) in species.items():
        sub = traj.atom_slice(idx)[picks]
        sasa_cache[name] = _sasa_nm2(sub)

    for n, f in enumerate(picks):
        frame_terms = {}
        for name, (_, _, idx) in species.items():
            pos = traj.xyz[f][idx]
            e = evaluators[name].energies(pos)
            e["nonpolar_solvation"] = _GAMMA_KJ_PER_NM2 * float(sasa_cache[name][n])
            frame_terms[name] = e
        delta = {
            k: (frame_terms["complex"][k] - frame_terms["receptor"][k]
                - frame_terms["ligand"][k]) * _KJ_TO_KCAL
            for k in terms
        }
        for k, v in delta.items():
            terms[k].append(v)
        per_frame.append(sum(delta.values()))

    components = {k: float(np.mean(v)) for k, v in terms.items()}
    prov = base_provenance(
        stage="mmgbsa", method="MM-GBSA (GBn2, single trajectory)",
        forcefield=ff_used, ligand_ff=ligand_ff, ligand_smiles=lig_smiles,
        n_frames_scored=len(picks), n_frames_total=int(traj.n_frames),
        frames=frames, md_refine_ns=md_refine_ns,
        nonpolar_gamma_kcal_mol_A2=0.0072, entropy_term=False,
        platform=plat.getName(), precision=precision, seed=seed,
        trajectory=traj_obj.traj_path, topology=traj_obj.top_path,
        trajectory_provenance=traj_obj.provenance,
    )

    result = MMGBSAResult(per_frame, components, n_frames=len(picks),
                          provenance=prov)
    if verbose:
        print(f"   dG = {result.dG:.2f} +/- {result.sem:.2f} kcal/mol "
              f"(vdW {components['vdw']:.1f}, elec "
              f"{components['electrostatic']:.1f}, polar "
              f"{components['polar_solvation']:.1f}, nonpolar "
              f"{components['nonpolar_solvation']:.1f})")
    return result


@register_function(
    aliases=["alchemical_free_energy", "fep", "ti", "自由能微扰",
             "absolute_binding_free_energy"],
    category="mol",
    description=(
        "Not implemented — calling this always raises NotImplementedError. "
        "It is registered so that a search for FEP / TI / absolute binding "
        "free energy says so outright and points somewhere useful, rather "
        "than returning nothing. Rigorous alchemical free energies need "
        "openmmtools for the alchemical factory plus many lambda windows — "
        "hundreds of GPU-hours per ligand. Use mmgbsa for fast relative "
        "ranking, or perses / OpenFE for a real FEP workflow."
    ),
    examples=["# not implemented; raises NotImplementedError. Use ov.mol.mmgbsa"],
    related=["mol.mmgbsa", "mol.simulate"],
)
def alchemical_free_energy(*args, **kwargs):
    r"""Alchemical (FEP/TI) binding free energy — **not implemented**."""
    raise NotImplementedError(
        "Alchemical free-energy calculations (FEP / TI) are not implemented "
        "in ov.mol.\n"
        "They are the rigorous way to get absolute or relative binding free "
        "energies, but need `openmmtools` for the alchemical factory plus "
        "many lambda windows — hundreds of GPU-hours per ligand, and careful "
        "convergence checking.\n"
        "For fast relative ranking use ov.mol.mmgbsa(); for a production FEP "
        "workflow see openmmtools / perses / OpenFE.")
