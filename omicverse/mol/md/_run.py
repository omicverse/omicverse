r"""The dynamics engine for :mod:`omicverse.mol.md`.

Three explicit stages — :func:`minimize`, :func:`equilibrate`,
:func:`production` — plus :func:`simulate`, the one-call orchestration that
covers the common case.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..._registry import register_function
from ._common import (base_provenance, check_md, explicit_device_requested,
                      file_hash, is_device_error, resolve_platform)
from ._types import MDSystem, MDTrajectory

# atom names that define the protein backbone
_BACKBONE = {"N", "CA", "C", "O"}


_GPU_RUNTIME_HINT = (
    "The {name} platform initialised but failed at runtime:\n"
    "  {err}\n"
    "The usual cause is a toolkit/driver mismatch — the OpenMM build was "
    "compiled with a newer CUDA than the installed driver supports "
    "(CUDA_ERROR_UNSUPPORTED_PTX_VERSION), or no device binary matches this "
    "GPU architecture.\n"
    "Fixes:\n"
    "  * pin the CUDA toolkit to your driver (check `nvidia-smi`), e.g.\n"
    "      conda install -c conda-forge 'cuda-version=12.4'\n"
    "  * or select another device: platform='OpenCL' / platform='CPU', or the "
    "OMICOS_MD_DEVICE environment variable."
)


def _build_simulation(mdsys: MDSystem, integrator_factory, platform: str,
                      precision: str, verbose: bool = True):
    """Instantiate an OpenMM ``Simulation`` on the resolved platform.

    ``integrator_factory`` is a zero-argument callable returning a *fresh*
    integrator — an ``Integrator`` cannot be reused across Contexts, so a new
    one is needed for each attempt when falling back between platforms.

    A platform can pass :func:`resolve_platform` (the shared library loads)
    and still fail when it compiles kernels for the actual GPU. In ``auto``
    mode that failure is caught and the next platform is tried; when a device
    was requested explicitly the error is re-raised with a diagnosis instead
    of silently running somewhere else.
    """
    import warnings

    from openmm import app

    explicit = explicit_device_requested(platform)

    def _make(plat, props):
        simulation = app.Simulation(mdsys.topology, mdsys.system,
                                    integrator_factory(), plat, props)
        simulation.context.setPositions(mdsys.positions)
        if mdsys.box_vectors is not None:
            simulation.context.setPeriodicBoxVectors(*mdsys.box_vectors)
        # touch the context so kernel compilation happens here, not later
        simulation.context.getState(getEnergy=True)
        return simulation

    plat, props = resolve_platform(platform, precision, verbose=verbose)
    try:
        return _make(plat, props)
    except Exception as exc:
        name = plat.getName()
        if not is_device_error(exc):
            raise
        hint = _GPU_RUNTIME_HINT.format(name=name, err=exc)
        if explicit or name == "CPU":
            raise RuntimeError(hint) from exc
        warnings.warn(
            f"{name} platform failed at runtime; falling back to a slower "
            f"device.\n{hint}", RuntimeWarning, stacklevel=2)

    for fallback in ("OpenCL", "CPU"):
        if fallback == plat.getName():
            continue
        try:
            plat_fb, props_fb = resolve_platform(fallback, precision,
                                                 verbose=verbose)
            return _make(plat_fb, props_fb)
        except Exception:
            continue
    raise RuntimeError(
        "no OpenMM platform could run this system — CUDA, OpenCL and CPU all "
        "failed. Check the OpenMM installation with "
        "`python -m openmm.testInstallation`.")


def _sync_back(mdsys: MDSystem, simulation) -> None:
    """Copy positions (and box) from a finished Simulation back into MDSystem."""
    state = simulation.context.getState(getPositions=True,
                                        enforcePeriodicBox=False)
    mdsys.positions = state.getPositions()
    if mdsys.box_vectors is not None:
        mdsys.box_vectors = state.getPeriodicBoxVectors()


def _restrained_atoms(topology, selection: str):
    """Indices to position-restrain: ``'heavy'``, ``'backbone'`` or ``'none'``."""
    sel = str(selection).lower()
    if sel == "none":
        return []
    solvent = {"HOH", "WAT", "NA", "CL", "K", "MG", "CA", "ZN", "SOD", "CLA"}
    out = []
    for atom in topology.atoms():
        if atom.residue.name in solvent:
            continue
        if atom.element is None or atom.element.symbol == "H":
            continue                                  # never restrain hydrogens
        if sel == "backbone" and atom.name not in _BACKBONE:
            continue
        out.append(atom.index)
    return out


def _add_position_restraints(system, topology, positions, selection: str,
                             k: float):
    """Add a harmonic position restraint force; return its force index."""
    import openmm
    from openmm import unit

    indices = _restrained_atoms(topology, selection)
    if not indices:
        return None
    force = openmm.CustomExternalForce(
        "k_restraint*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter(
        "k_restraint", k * unit.kilojoule_per_mole / unit.nanometer**2)
    for p in ("x0", "y0", "z0"):
        force.addPerParticleParameter(p)
    pos = positions.value_in_unit(unit.nanometer)
    for i in indices:
        force.addParticle(int(i), [pos[i][0], pos[i][1], pos[i][2]])
    return system.addForce(force)


def _remove_forces(system, indices) -> None:
    """Remove forces by index, high-to-low so earlier indices stay valid."""
    for idx in sorted([i for i in indices if i is not None], reverse=True):
        system.removeForce(idx)


def _check_timestep(mdsys: MDSystem, timestep_fs: float) -> None:
    """Guard the classic blow-up: a long timestep without hydrogen-mass repartitioning."""
    hmr = bool(mdsys.provenance.get("hmr", False))
    if timestep_fs > 2.5 and not hmr:
        raise ValueError(
            f"timestep_fs={timestep_fs} needs hydrogen-mass repartitioning to "
            "be stable. Rebuild the system with "
            "prepare_system(..., hmr=True) (or simulate(..., hmr=True)), or "
            "drop to timestep_fs=2.0.")
    if timestep_fs > 5.0:
        raise ValueError(
            f"timestep_fs={timestep_fs} is beyond what HMR + HBonds "
            "constraints can integrate stably; 4 fs is the practical maximum.")


@register_function(
    aliases=["minimize", "energy_minimization", "能量最小化", "minimise",
             "md_minimize", "最小化"],
    category="mol",
    description=(
        "Energy-minimise a prepared MDSystem with OpenMM's L-BFGS minimiser, "
        "relieving steric clashes left by structure repair, ligand placement "
        "and solvation. Always run this before dynamics — an unminimised "
        "system is the most common cause of 'Particle coordinate is NaN' "
        "blow-ups. Prints the potential energy before and after, and updates "
        "the system's coordinates in place."
    ),
    examples=[
        "sys = ov.mol.minimize(sys)",
        "sys = ov.mol.minimize(sys, max_iterations=5000)",
    ],
    related=["mol.prepare_system", "mol.equilibrate", "mol.production",
             "mol.simulate"],
)
def minimize(system: MDSystem, *, max_iterations: int = 0,
             tolerance_kj_mol_nm: float = 10.0, platform: str = "auto",
             precision: str = "mixed", verbose: bool = True) -> MDSystem:
    r"""Minimise the potential energy of ``system`` (coordinates updated in place).

    Parameters
    ----------
    system
        A system from :func:`~omicverse.mol.prepare_system`.
    max_iterations
        Minimiser iteration cap; ``0`` means "run to convergence".
    tolerance_kj_mol_nm
        Convergence tolerance on the force, kJ/mol/nm.
    platform, precision
        Compute device — see :func:`~omicverse.mol.production`.

    Returns
    -------
    MDSystem
        The same object, with minimised ``positions``.
    """
    check_md()
    import openmm
    from openmm import unit

    def _integrator():
        return openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)

    simulation = _build_simulation(system, _integrator, platform, precision,
                                   verbose=verbose)

    e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    simulation.minimizeEnergy(
        maxIterations=int(max_iterations),
        tolerance=tolerance_kj_mol_nm * unit.kilojoule_per_mole / unit.nanometer)
    e1 = simulation.context.getState(getEnergy=True).getPotentialEnergy()

    if verbose:
        print(f"   minimisation: {e0.value_in_unit(unit.kilojoule_per_mole):.1f}"
              f" -> {e1.value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol")

    _sync_back(system, simulation)
    system.provenance.setdefault("stages", []).append({
        "stage": "minimize",
        "energy_before_kj_mol": float(e0.value_in_unit(unit.kilojoule_per_mole)),
        "energy_after_kj_mol": float(e1.value_in_unit(unit.kilojoule_per_mole)),
        "max_iterations": int(max_iterations),
        "platform": simulation.context.getPlatform().getName(),
    })
    del simulation
    return system


@register_function(
    aliases=["equilibrate", "equilibration", "平衡", "nvt_npt",
             "md_equilibrate", "体系平衡"],
    category="mol",
    description=(
        "Equilibrate a minimised MDSystem in two stages: NVT (heat to the "
        "target temperature at fixed volume) then NPT (relax the box density "
        "under a Monte-Carlo barostat). Heavy atoms are position-restrained "
        "and the restraint is released linearly over the NPT stage, so the "
        "structure settles without being torn apart. Skipping equilibration "
        "gives production runs with the wrong density and inflated early "
        "RMSD. Updates coordinates and box vectors in place."
    ),
    examples=[
        "sys = ov.mol.equilibrate(sys)",
        "sys = ov.mol.equilibrate(sys, nvt_ps=200, npt_ps=500, restraint='backbone')",
    ],
    related=["mol.minimize", "mol.production", "mol.simulate",
             "mol.prepare_system"],
)
def equilibrate(system: MDSystem, *, nvt_ps: float = 100.0,
                npt_ps: float = 100.0, temperature_K: float = 300.0,
                pressure_bar: float = 1.0, restraint: str = "heavy",
                restraint_k: float = 1000.0, timestep_fs: float = 2.0,
                friction_per_ps: float = 1.0, platform: str = "auto",
                precision: str = "mixed", seed: int = 0,
                progress: bool = True, verbose: bool = True) -> MDSystem:
    r"""Heat and density-equilibrate a system (coordinates updated in place).

    Parameters
    ----------
    nvt_ps, npt_ps
        Length of the constant-volume and constant-pressure stages, in ps.
        ``npt_ps=0`` is forced for implicit solvent (no box to relax).
    temperature_K, pressure_bar
        Thermostat and barostat targets.
    restraint
        Which atoms to position-restrain during NVT: ``'heavy'`` (all
        non-hydrogen solute atoms), ``'backbone'``, or ``'none'``.
    restraint_k
        Restraint force constant, kJ/mol/nm^2, released linearly to 0 across
        the NPT stage.
    seed
        Seed for the initial Maxwell-Boltzmann velocities and the barostat.
    progress
        Accepted for signature symmetry with
        :func:`~omicverse.mol.production`, but **not used**: equilibration
        attaches no reporters, so there is no per-step progress to suppress.
        Use ``verbose`` to control the stage banners.

    Returns
    -------
    MDSystem
        The same object, equilibrated.
    """
    check_md()
    import openmm
    from openmm import unit

    _check_timestep(system, timestep_fs)
    implicit = not system.is_periodic
    if implicit and npt_ps:
        if verbose:
            print("   implicit solvent: skipping the NPT stage (no box)")
        npt_ps = 0.0

    added: list = []
    restraint_idx = None
    if str(restraint).lower() != "none":
        restraint_idx = _add_position_restraints(
            system.system, system.topology, system.positions, restraint,
            restraint_k)
        added.append(restraint_idx)

    barostat_idx = None
    if npt_ps > 0:
        barostat = openmm.MonteCarloBarostat(pressure_bar * unit.bar,
                                             temperature_K * unit.kelvin, 25)
        barostat.setRandomNumberSeed(int(seed))
        barostat_idx = system.system.addForce(barostat)
        added.append(barostat_idx)

    def _integrator():
        integ = openmm.LangevinMiddleIntegrator(
            temperature_K * unit.kelvin, friction_per_ps / unit.picosecond,
            timestep_fs * unit.femtoseconds)
        integ.setRandomNumberSeed(int(seed))
        return integ

    simulation = _build_simulation(system, _integrator, platform, precision,
                                   verbose=verbose)
    simulation.context.setVelocitiesToTemperature(
        temperature_K * unit.kelvin, int(seed))

    steps_per_ps = int(round(1000.0 / timestep_fs))
    nvt_steps = int(nvt_ps * steps_per_ps)
    npt_steps = int(npt_ps * steps_per_ps)

    try:
        if nvt_steps:
            if verbose:
                print(f"   NVT {nvt_ps:g} ps @ {temperature_K:g} K "
                      f"(restraint '{restraint}', k={restraint_k:g})")
            simulation.step(nvt_steps)

        if npt_steps:
            if verbose:
                print(f"   NPT {npt_ps:g} ps @ {pressure_bar:g} bar "
                      "(restraint released linearly)")
            if restraint_idx is not None:
                # release the restraint over 10 equal windows
                n_windows = 10
                per = max(npt_steps // n_windows, 1)
                for w in range(n_windows):
                    # setParameter takes a bare number in OpenMM's internal
                    # units, which for this force constant is kJ/mol/nm^2.
                    k_now = restraint_k * (1.0 - (w + 1) / n_windows)
                    simulation.context.setParameter("k_restraint", k_now)
                    simulation.step(per)
                done = per * n_windows
                if npt_steps > done:
                    simulation.step(npt_steps - done)
            else:
                simulation.step(npt_steps)

        state = simulation.context.getState(getEnergy=True)
        e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if e != e:  # NaN
            raise RuntimeError(
                "the system blew up during equilibration (energy is NaN). "
                "Usual causes: skipped minimisation, a badly placed ligand, or "
                "steric clashes left by structure repair. Re-run "
                "prepare_system and minimize first, and check the clash "
                "warning.")
        if verbose:
            print(f"   equilibrated: potential energy {e:.1f} kJ/mol")

        _sync_back(system, simulation)
    finally:
        _remove_forces(system.system, added)
        del simulation

    system.provenance.setdefault("stages", []).append({
        "stage": "equilibrate", "nvt_ps": nvt_ps, "npt_ps": npt_ps,
        "temperature_K": temperature_K, "pressure_bar": pressure_bar,
        "restraint": restraint, "restraint_k": restraint_k,
        "timestep_fs": timestep_fs, "seed": seed,
    })
    return system


@register_function(
    aliases=["production", "production_md", "产出模拟", "md_production",
             "run_md", "生产模拟"],
    category="mol",
    description=(
        "Run production molecular dynamics and write a trajectory. Uses a "
        "Langevin-middle integrator with HBonds constraints, NPT (Monte-Carlo "
        "barostat) or NVT, writing DCD frames, a state log (energy, "
        "temperature, volume, ns/day) and periodic checkpoints. Runs on GPU "
        "via the CUDA platform when available and prints which platform it "
        "actually used. Supports restarting from a checkpoint. Returns an "
        "MDTrajectory handle for the analysis functions."
    ),
    examples=[
        "traj = ov.mol.production(sys, ns=10)",
        "traj = ov.mol.production(sys, ns=100, timestep_fs=4.0)  # needs hmr=True",
        "traj = ov.mol.production(sys, ns=10, resume='md_out/prod.chk')",
    ],
    related=["mol.simulate", "mol.equilibrate", "mol.rmsd", "mol.mmgbsa"],
)
def production(system: MDSystem, *, ns: float = 10.0,
               timestep_fs: float = 2.0, ensemble: str = "NPT",
               temperature_K: float = 300.0, pressure_bar: float = 1.0,
               report_ps: float = 10.0, traj: str = "prod.dcd",
               log: Optional[str] = None, checkpoint_ps: float = 1000.0,
               resume: Optional[str] = None, friction_per_ps: float = 1.0,
               platform: str = "auto", precision: str = "mixed",
               seed: int = 0, progress: bool = True,
               update_system: bool = False,
               verbose: bool = True) -> MDTrajectory:
    r"""Run production dynamics and write a trajectory.

    Parameters
    ----------
    ns
        Simulated time, in nanoseconds.
    timestep_fs
        Integration timestep. 2 fs with HBonds constraints is the safe
        default; 4 fs requires a system built with ``hmr=True``.
    ensemble
        ``'NPT'`` (constant pressure) or ``'NVT'``. NPT needs a periodic box,
        so on an implicit-solvent system it is downgraded to NVT rather than
        refused — the run still happens, and the ensemble actually used is the
        one recorded in the trajectory's provenance.
    report_ps
        Interval between saved frames and log lines, in ps.
    traj
        Output trajectory path (``.dcd``; ``.xtc`` if OpenMM provides a
        writer).
    log
        State-log path; defaults to the trajectory path with a ``.log``
        suffix.
    checkpoint_ps
        Interval between binary checkpoints (for ``resume``).
    resume
        Path to a checkpoint to continue from.
    platform
        ``'auto'`` (CUDA → OpenCL → CPU), or a specific platform name. The
        ``OMICOS_MD_DEVICE`` environment variable overrides this.
    precision
        ``'mixed'`` (default), ``'single'`` or ``'double'`` on GPU platforms.
    seed
        Seed for the initial velocities, thermostat and barostat.
    update_system
        Write the final coordinates back into ``system``. Off by default so
        that repeated calls are independent replicas rather than a chained
        continuation; use ``resume=`` to continue a run.

    Returns
    -------
    MDTrajectory

    Notes
    -----
    **Reproducibility.** A run repeats exactly when the ``MDSystem``, ``seed``,
    ``platform`` and ``precision`` all match — verified bit-for-bit on both the
    CPU and CUDA platforms. Because this function does not mutate ``system``,
    calling it twice yields two identical replicas rather than a continuation;
    use ``resume=`` to continue a run, or ``update_system=True`` to chain.

    Across **different** platforms or precisions trajectories do diverge: MD is
    chaotic, so tiny floating-point differences grow exponentially. That is
    physics, not a bug — compare ensemble properties (RMSD/Rg distributions,
    free energies), not individual frames.
    """
    check_md()
    import openmm
    from openmm import app, unit

    _check_timestep(system, timestep_fs)
    ens = str(ensemble).upper()
    if ens not in ("NPT", "NVT"):
        raise ValueError(f"ensemble must be 'NPT' or 'NVT', got {ensemble!r}")
    if ens == "NPT" and not system.is_periodic:
        if verbose:
            print("   implicit solvent has no box — running NVT instead of NPT")
        ens = "NVT"

    outdir = os.path.dirname(os.path.abspath(traj))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    top_path = os.path.splitext(traj)[0] + "_topology.pdb"
    log_path = log or (os.path.splitext(traj)[0] + ".log")
    chk_path = os.path.splitext(traj)[0] + ".chk"

    steps_per_ps = 1000.0 / timestep_fs
    n_steps = int(round(ns * 1000.0 * steps_per_ps))
    report_interval = max(int(round(report_ps * steps_per_ps)), 1)
    chk_interval = max(int(round(checkpoint_ps * steps_per_ps)), report_interval)
    if n_steps < 1:
        raise ValueError(
            f"ns={ns} with timestep_fs={timestep_fs} gives zero steps; "
            "increase ns.")

    barostat_idx = None
    if ens == "NPT":
        barostat = openmm.MonteCarloBarostat(pressure_bar * unit.bar,
                                             temperature_K * unit.kelvin, 25)
        barostat.setRandomNumberSeed(int(seed))
        barostat_idx = system.system.addForce(barostat)

    def _integrator():
        integ = openmm.LangevinMiddleIntegrator(
            temperature_K * unit.kelvin, friction_per_ps / unit.picosecond,
            timestep_fs * unit.femtoseconds)
        integ.setRandomNumberSeed(int(seed))
        return integ

    simulation = _build_simulation(system, _integrator, platform, precision,
                                   verbose=verbose)
    plat_name = simulation.context.getPlatform().getName()

    # topology for the analysis layer
    with open(top_path, "w") as fh:
        app.PDBFile.writeFile(system.topology, system.positions, fh,
                              keepIds=True)

    if resume:
        if not os.path.exists(resume):
            raise FileNotFoundError(f"checkpoint not found: {resume}")
        simulation.loadCheckpoint(resume)
        if verbose:
            print(f"   resumed from checkpoint {resume}")
    else:
        simulation.context.setVelocitiesToTemperature(
            temperature_K * unit.kelvin, int(seed))

    # reporters: trajectory, human-readable log, checkpoint
    append = bool(resume) and os.path.exists(traj)
    if traj.lower().endswith(".xtc"):
        if not hasattr(app, "XTCReporter"):
            raise ValueError(
                "this OpenMM build has no XTCReporter; use a '.dcd' path "
                "(MDTraj reads both).")
        simulation.reporters.append(
            app.XTCReporter(traj, report_interval, append=append))
    else:
        simulation.reporters.append(
            app.DCDReporter(traj, report_interval, append=append))

    state_kw: Dict[str, Any] = dict(
        step=True, time=True, potentialEnergy=True, temperature=True,
        speed=True, totalSteps=n_steps, remainingTime=True, separator="\t")
    if system.is_periodic:
        state_kw["volume"] = True
        state_kw["density"] = True
    simulation.reporters.append(
        app.StateDataReporter(log_path, report_interval, **state_kw))
    if progress:
        import sys as _sys
        simulation.reporters.append(
            app.StateDataReporter(_sys.stdout, max(n_steps // 10, 1),
                                  **state_kw))
    simulation.reporters.append(
        app.CheckpointReporter(chk_path, chk_interval))

    if verbose:
        print(f"   production: {ns:g} ns, {n_steps} steps @ {timestep_fs:g} fs, "
              f"{ens}, frame every {report_ps:g} ps -> {traj}")

    try:
        simulation.step(n_steps)
    finally:
        if barostat_idx is not None:
            _remove_forces(system.system, [barostat_idx])

    state = simulation.context.getState(getEnergy=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if e != e:
        raise RuntimeError(
            "the simulation blew up (potential energy is NaN). Re-run with a "
            "shorter timestep, ensure the system was minimised and "
            "equilibrated, and check for a badly parameterised ligand.")

    # Deliberately do NOT write the final state back into `system` by default.
    # production() produces a *trajectory*; mutating its input would make a
    # second call silently start from the first run's endpoint — breaking both
    # reproducibility and the obvious "run two replicas from this system"
    # idiom. Continue a run with resume=<checkpoint> instead.
    if update_system:
        _sync_back(system, simulation)
    simulation.saveCheckpoint(chk_path)
    n_frames = n_steps // report_interval

    prov = base_provenance(
        stage="production",
        integrator="LangevinMiddle", ensemble=ens,
        timestep_fs=timestep_fs, temperature_K=temperature_K,
        pressure_bar=pressure_bar if ens == "NPT" else None,
        friction_per_ps=friction_per_ps, constraints="HBonds",
        ns=ns, n_steps=n_steps, report_ps=report_ps,
        platform=plat_name, precision=precision, seed=seed,
        resumed_from=resume,
        forcefield=system.forcefield, water=system.water,
        system_provenance=system.provenance,
        files={"trajectory": traj, "trajectory_sha256": file_hash(traj),
               "topology": top_path, "log": log_path, "checkpoint": chk_path},
    )
    if verbose:
        print(f"   done: {n_frames} frames on platform {plat_name}")

    del simulation
    return MDTrajectory(traj, top_path, n_frames=n_frames, dt_ps=report_ps,
                        checkpoint=chk_path, log_path=log_path,
                        provenance=prov)


@register_function(
    aliases=["simulate", "molecular_dynamics", "分子动力学", "md", "openmm",
             "run_simulation", "跑动力学", "动力学模拟"],
    category="mol",
    description=(
        "Run a complete molecular-dynamics simulation from a static "
        "structure in one call: prepare (repair + parameterise + solvate), "
        "minimise, equilibrate (NVT then NPT), and produce a trajectory. "
        "Accepts a MolStructure, a PDB file/identifier, or a DockingResult — "
        "so an AlphaFold model or a docked pose can be taken straight to "
        "dynamics. GPU-accelerated through OpenMM's CUDA platform when "
        "available (prints the platform and ns/day). Returns an MDTrajectory "
        "for rmsd / rmsf / hbonds / mmgbsa analysis. For step-by-step "
        "control use prepare_system + minimize + equilibrate + production."
    ),
    examples=[
        "traj = ov.mol.simulate(ov.mol.fetch_structure('1UBQ'), ns=10)",
        "traj = ov.mol.simulate(s, ns=5, water='implicit')  # fast, no box",
        "traj = ov.mol.simulate(docking_result, ns=5)       # dock -> MD",
        "ov.mol.plot_rmsd(traj)",
    ],
    related=["mol.prepare_system", "mol.production", "mol.mmgbsa",
             "mol.dock", "mol.rmsd", "mol.fetch_structure"],
)
def simulate(
    structure,
    *,
    ns: float = 10.0,
    ligand=None,
    minimize_first: bool = True,
    equilibrate_first: bool = True,
    forcefield: str = "amber14-all",
    water: str = "tip3p",
    box_padding_nm: float = 1.0,
    ions: str = "0.15 M NaCl",
    ph: float = 7.0,
    ligand_ff: str = "gaff-2.11",
    temperature_K: float = 300.0,
    pressure_bar: float = 1.0,
    timestep_fs: float = 2.0,
    hmr: bool = False,
    ensemble: str = "NPT",
    nvt_ps: float = 100.0,
    npt_ps: float = 100.0,
    report_ps: float = 10.0,
    outdir: str = "md_out",
    traj_name: str = "prod.dcd",
    platform: str = "auto",
    precision: str = "mixed",
    seed: int = 0,
    verbose: bool = True,
) -> MDTrajectory:
    r"""Prepare, minimise, equilibrate and run production MD in one call.

    Parameters
    ----------
    structure
        A :class:`~omicverse.mol.MolStructure`, a PDB file or identifier, or
        a :class:`~omicverse.mol.DockingResult` (receptor + top pose).
    ns
        Production length, in nanoseconds.
    ligand
        Optional small molecule to co-simulate (SMILES / ``.sdf`` / RDKit
        ``Mol`` / ``DockingResult``).
    minimize_first, equilibrate_first
        Stages to run before production. Leave both on unless you know the
        input is already relaxed — skipping minimisation is the usual cause
        of a NaN blow-up.
    water
        ``'tip3p'`` etc., or ``'implicit'`` for fast GBn2 implicit solvent.
    hmr, timestep_fs
        Hydrogen-mass repartitioning allows ``timestep_fs=4.0``.
    outdir
        Directory for the trajectory, topology, log and checkpoint.

    Returns
    -------
    MDTrajectory

    Notes
    -----
    Fixed ``seed`` + same platform + same precision reproduces a run exactly
    (bit-for-bit on both CPU and CUDA); across platforms or precisions the
    trajectory diverges, because MD is chaotic. ``seed`` also controls the
    stochastic parts of preparation — hydrogen placement and water/ion
    positions — so the whole pipeline is reproducible, not just the dynamics.
    """
    from ._system import prepare_system

    os.makedirs(outdir, exist_ok=True)
    if verbose:
        print(f"🧬 ov.mol.simulate — {ns:g} ns, output in {outdir}/")

    mdsys = prepare_system(
        structure, forcefield=forcefield, water=water,
        box_padding_nm=box_padding_nm, ions=ions, ph=ph, ligand=ligand,
        ligand_ff=ligand_ff, hmr=hmr, seed=seed, verbose=verbose)

    if minimize_first:
        mdsys = minimize(mdsys, platform=platform, precision=precision,
                         verbose=verbose)
    if equilibrate_first:
        mdsys = equilibrate(
            mdsys, nvt_ps=nvt_ps, npt_ps=npt_ps, temperature_K=temperature_K,
            pressure_bar=pressure_bar, timestep_fs=timestep_fs,
            platform=platform, precision=precision, seed=seed,
            verbose=verbose)

    return production(
        mdsys, ns=ns, timestep_fs=timestep_fs, ensemble=ensemble,
        temperature_K=temperature_K, pressure_bar=pressure_bar,
        report_ps=report_ps, traj=os.path.join(outdir, traj_name),
        platform=platform, precision=precision, seed=seed, verbose=verbose)
