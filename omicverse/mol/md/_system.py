r"""System preparation for :mod:`omicverse.mol.md`.

Turns a static structure (a :class:`~omicverse.mol.MolStructure`, a PDB file,
a PDB/gene identifier, or a :class:`~omicverse.mol.DockingResult` pose) into a
solvated, parameterised, simulatable :class:`~omicverse.mol.md.MDSystem`.
"""

from __future__ import annotations

import contextlib
import os
import random as _random
import re
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..._registry import register_function
from ._common import base_provenance, check_md, file_hash, need_md
from ._types import MDSystem

# protein forcefield -> main XML
_FF_FILES = {
    "amber14-all": "amber14-all.xml",
    "amber14": "amber14-all.xml",
    "amber99sb": "amber99sbildn.xml",
    "charmm36": "charmm36.xml",
}

# (forcefield, water) -> water XML
_WATER_FILES = {
    ("amber14-all", "tip3p"): "amber14/tip3p.xml",
    ("amber14-all", "tip3pfb"): "amber14/tip3pfb.xml",
    ("amber14-all", "tip4pew"): "amber14/tip4pew.xml",
    ("amber14-all", "spce"): "amber14/spce.xml",
    ("amber99sb", "tip3p"): "amber99_obc.xml",
    ("charmm36", "tip3p"): "charmm36/water.xml",
    ("charmm36", "tip4pew"): "charmm36/tip4pew.xml",
    ("charmm36", "spce"): "charmm36/spce.xml",
}

# OpenMM Modeller.addSolvent water model names
_SOLVENT_MODELS = {"tip3p": "tip3p", "tip3pfb": "tip3p", "tip4pew": "tip4pew",
                   "spce": "spce", "tip5p": "tip5p"}

_IMPLICIT_XML = "implicit/gbn2.xml"

# Residue name given to a parameterised small molecule. Fixed so that the
# natural MDTraj selection "resname LIG" works for every ov.mol complex.
LIGAND_RESNAME = "LIG"


@contextlib.contextmanager
def _seeded_rng(seed: int):
    """Temporarily seed the global RNGs, restoring the caller's state after.

    OpenMM's ``Modeller.addHydrogens`` and ``Modeller.addSolvent`` draw from
    Python's global :mod:`random` (and NumPy's) rather than taking a seed
    argument, so hydrogen placement and water/ion positions vary between
    otherwise identical runs. Seeding here makes :func:`prepare_system`
    reproducible; saving and restoring the previous state keeps us from
    hijacking the RNG of whatever called us.
    """
    py_state = _random.getstate()
    np_state = np.random.get_state()
    _random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    try:
        yield
    finally:
        _random.setstate(py_state)
        np.random.set_state(np_state)


def _ambertools_path_hint() -> str:
    """Extra hint when AmberTools is installed but invisible on ``PATH``.

    The OpenFF toolkit detects AmberTools with :func:`shutil.which`, so an
    interpreter launched by absolute path (a Jupyter kernel, a cron job, a
    venv that was never "activated") can have the binaries sitting right next
    to it and still report them as missing. That failure is very confusing
    without this pointer.
    """
    import shutil
    import sys

    found = shutil.which("antechamber")
    if found:
        # On PATH but still failing: almost always AMBERHOME, which
        # antechamber needs to locate its parameter files (gaff.dat etc.).
        # Without it antechamber exits non-zero and OpenFF just reports
        # "no toolkit can assign_partial_charges".
        if not os.environ.get("AMBERHOME"):
            root = os.path.dirname(os.path.dirname(found))
            return (
                "NOTE: antechamber is on PATH but AMBERHOME is not set, so it "
                "cannot find its parameter files and exits with an error.\n"
                f"    export AMBERHOME=\"{root}\"\n")
        return ""

    local = os.path.join(os.path.dirname(sys.executable), "antechamber")
    if os.path.exists(local):
        return (
            "NOTE: antechamber IS installed at\n"
            f"    {local}\n"
            "but it is not on PATH, which is how the OpenFF toolkit looks for "
            "it. Activate the environment, or prepend it:\n"
            f"    export PATH=\"{os.path.dirname(local)}:$PATH\"\n"
            f"    export AMBERHOME=\"{os.path.dirname(os.path.dirname(local))}\"\n")
    return ""


def _norm_ff(forcefield: str) -> str:
    ff = str(forcefield).lower().strip()
    if ff not in _FF_FILES:
        raise ValueError(
            f"unknown forcefield {forcefield!r}; supported: "
            f"{sorted(set(_FF_FILES))}")
    return "amber14-all" if ff in ("amber14", "amber14-all") else ff


def _parse_ionic_strength(ions: str) -> Tuple[float, str, str]:
    """``'0.15 M NaCl'`` -> ``(0.15, 'Na+', 'Cl-')``; ``'none'`` -> ``(0.0, ...)``.

    Ions are always added to neutralise the net charge; the concentration only
    controls the *extra* salt on top of neutralisation.
    """
    if ions is None or str(ions).strip().lower() in ("none", "", "neutral"):
        return 0.0, "Na+", "Cl-"
    text = str(ions)
    m = re.search(r"([0-9]*\.?[0-9]+)\s*M", text, flags=re.I)
    conc = float(m.group(1)) if m else 0.15
    salt = re.sub(r"[0-9]*\.?[0-9]+\s*M", "", text, flags=re.I).strip()
    pairs = {"nacl": ("Na+", "Cl-"), "kcl": ("K+", "Cl-"),
             "nabr": ("Na+", "Br-"), "cacl2": ("Ca2+", "Cl-")}
    pos, neg = pairs.get(salt.lower().replace(" ", ""), ("Na+", "Cl-"))
    return conc, pos, neg


def _resolve_structure(structure, workdir: str) -> Tuple[str, Any, Dict[str, Any]]:
    """Normalise the ``structure`` argument to ``(pdb_path, ligand, meta)``.

    Accepts a :class:`MolStructure`, a path to a PDB file, a PDB/gene
    identifier (fetched on demand), or a :class:`DockingResult` — in which
    case the receptor **and** the top-ranked pose are pulled out so the
    complex can be simulated.
    """
    meta: Dict[str, Any] = {}
    ligand = None

    # DockingResult -> receptor + best pose (dock -> MD handoff)
    if hasattr(structure, "poses") and hasattr(structure, "affinities"):
        receptor = getattr(structure, "receptor", None)
        if receptor is None:
            # DockingResult.receptor arrived with the MD layer in 2.3.0, so a
            # result pickled by an earlier omicverse has no such attribute and
            # lands here. Re-docking is not required — the receptor is the only
            # missing piece, and it can simply be passed alongside the poses.
            raise ValueError(
                "this DockingResult carries no receptor, so the complex cannot "
                "be rebuilt. Pass the receptor as the structure and the result "
                "as the ligand:\n"
                "    ov.mol.simulate(structure, ligand=result)\n"
                "with `structure` the MolStructure you docked into. Results "
                "from ov.mol.dock() on omicverse >= 2.3.0 carry the receptor "
                "themselves; one saved by an older version does not, so either "
                "hand it over this way or re-run ov.mol.dock().")
        ligand = structure.best
        if ligand is None:
            raise ValueError("DockingResult has no poses to simulate")
        try:
            meta["docking_affinity_kcal_mol"] = float(structure.affinities[0])
        except Exception:
            pass
        meta["source"] = "DockingResult"
        pdb_path, _, sub = _resolve_structure(receptor, workdir)
        meta.update({k: v for k, v in sub.items() if k != "source"})
        return pdb_path, ligand, meta

    # MolStructure
    if hasattr(structure, "to_pdb") and hasattr(structure, "atoms"):
        path = os.path.join(workdir, "receptor_input.pdb")
        structure.to_pdb(path)
        meta.setdefault("source", getattr(structure, "source", "MolStructure"))
        meta["gene"] = getattr(structure, "gene", None)
        meta["uniprot"] = getattr(structure, "uniprot", None)
        return path, ligand, meta

    # path or identifier
    if isinstance(structure, str):
        if os.path.exists(structure):
            meta["source"] = "file"
            return structure, ligand, meta
        from .._structure import fetch_structure
        s = fetch_structure(structure)
        path = os.path.join(workdir, "receptor_input.pdb")
        s.to_pdb(path)
        meta["source"] = getattr(s, "source", "fetched")
        meta["query"] = structure
        return path, ligand, meta

    raise TypeError(
        "structure must be a MolStructure, a DockingResult, a path to a PDB "
        f"file, or a PDB/gene identifier — got {type(structure).__name__}")


def _to_openff_molecule(ligand, *, seed: int = 0):
    """Build an OpenFF ``Molecule`` (with 3D coordinates) from many inputs.

    Accepts a SMILES string, an ``.sdf`` / ``.mol`` / ``.mol2`` path, an RDKit
    ``Mol`` (e.g. a docked pose), or a :class:`DockingResult`.
    """
    Molecule = need_md("openff.toolkit", "ov.mol ligand parameterisation").Molecule

    # DockingResult -> its best pose
    if hasattr(ligand, "poses") and hasattr(ligand, "affinities"):
        ligand = ligand.best

    mol = None
    try:
        if hasattr(ligand, "GetNumAtoms"):                      # RDKit Mol
            from rdkit import Chem
            rd = Chem.Mol(ligand)
            if rd.GetNumConformers() == 0:
                from rdkit.Chem import AllChem
                rd = Chem.AddHs(rd)
                AllChem.EmbedMolecule(rd, randomSeed=seed or 42)
                AllChem.MMFFOptimizeMolecule(rd)
            else:
                rd = Chem.AddHs(rd, addCoords=True)
            mol = Molecule.from_rdkit(rd, allow_undefined_stereo=True)
        elif isinstance(ligand, str) and os.path.exists(ligand):
            mol = Molecule.from_file(ligand, allow_undefined_stereo=True)
            if isinstance(mol, list):
                mol = mol[0]
        elif isinstance(ligand, str):
            mol = Molecule.from_smiles(ligand, allow_undefined_stereo=True)
        elif hasattr(ligand, "to_topology"):                    # already OpenFF
            mol = ligand
    except Exception as exc:
        raise ValueError(
            f"could not build a ligand molecule from {type(ligand).__name__}: "
            f"{exc}\nSupported: SMILES string, .sdf/.mol/.mol2 file, RDKit Mol, "
            "or a DockingResult.") from exc

    if mol is None:
        raise TypeError(
            f"unsupported ligand type {type(ligand).__name__}; pass a SMILES "
            "string, a .sdf/.mol2 path, an RDKit Mol, or a DockingResult")

    if mol.n_conformers == 0:
        mol.generate_conformers(n_conformers=1)
    return mol


def _assign_charges(off_mol, *, seed: int = 0, verbose: bool = False):
    """Give ``off_mol`` AM1-BCC partial charges, computed from a clean conformer.

    AM1-BCC runs a semi-empirical QM optimisation (``sqm``) whose convergence
    depends on the input geometry. A **docked pose** is a poor starting point —
    AutoDock assembles it from rigid fragments and hydrogens are added
    afterwards — and ``sqm`` can grind for many minutes on it before giving up,
    surfacing only as the opaque "no registered toolkits can provide
    assign_partial_charges".

    Partial charges are a property of the *molecule*, not of the pose, so they
    are computed from a freshly generated conformer and transferred back. This
    is the standard practice, it is far faster, and the docked coordinates are
    left untouched — those are what actually gets simulated.
    """
    import copy

    clean = copy.deepcopy(off_mol)
    clean._conformers = None
    clean.generate_conformers(n_conformers=1)
    clean.assign_partial_charges("am1bcc")
    off_mol.partial_charges = clean.partial_charges
    if verbose:
        print("   AM1-BCC charges assigned (from a generated conformer)")
    return off_mol


def _register_ligand_template(forcefield, off_mol, ligand_ff: str):
    """Attach a GAFF or SMIRNOFF residue-template generator for ``off_mol``."""
    generators = need_md("openmmforcefields.generators",
                         "ov.mol ligand parameterisation")
    ff = str(ligand_ff).lower()
    try:
        if ff.startswith("gaff"):
            gen = generators.GAFFTemplateGenerator(molecules=[off_mol],
                                                   forcefield=ligand_ff)
        elif ff.startswith("openff") or ff.startswith("smirnoff"):
            gen = generators.SMIRNOFFTemplateGenerator(molecules=[off_mol],
                                                       forcefield=ligand_ff)
        else:
            raise ValueError(
                f"unknown ligand_ff {ligand_ff!r}; use e.g. 'gaff-2.11' or "
                "'openff-2.1.0'")
        forcefield.registerTemplateGenerator(gen.generator)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"could not parameterise the ligand with {ligand_ff!r}: {exc}\n"
            "Small-molecule parameterisation fails for unusual chemistry "
            "(covalent warheads, metal coordination, exotic elements). Try the "
            "other backend (ligand_ff='openff-2.1.0' vs 'gaff-2.11'), or supply "
            "an already-parameterised system.") from exc


def _clash_report(topology, positions, *, cutoff_nm: float = 0.08) -> int:
    """Count non-bonded atom pairs closer than ``cutoff_nm`` (steric clashes).

    A handful is normal after solvation; hundreds means the structure needs
    fixing before dynamics, and minimisation will likely blow up.
    """
    from openmm import unit
    try:
        pos = np.asarray(positions.value_in_unit(unit.nanometer))
    except Exception:
        return 0
    if len(pos) > 60000:            # skip the O(N log N) tree on huge systems
        return 0
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return 0
    bonded = set()
    for bond in topology.bonds():
        i, j = bond[0].index, bond[1].index
        bonded.add((min(i, j), max(i, j)))
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=cutoff_nm)
    return sum(1 for (i, j) in pairs if (min(i, j), max(i, j)) not in bonded)


@register_function(
    aliases=["prepare_system", "md_prepare", "体系准备", "solvate",
             "build_system", "溶剂化", "md_system"],
    category="mol",
    description=(
        "Build a simulatable molecular-dynamics system from a static "
        "structure: repair missing atoms/residues and add hydrogens at a "
        "given pH (PDBFixer), optionally parameterise a small-molecule "
        "ligand (GAFF / OpenFF), solvate in a periodic water box with "
        "neutralising salt (or use GBn2 implicit solvent), and build the "
        "OpenMM System with the chosen protein forcefield. Accepts a "
        "MolStructure, a PDB file/identifier, or a DockingResult (whose top "
        "pose is rebuilt as a protein-ligand complex). Returns an MDSystem "
        "carrying topology, system, positions and full provenance; feed it "
        "to minimize / equilibrate / production."
    ),
    examples=[
        "sys = ov.mol.prepare_system(ov.mol.fetch_structure('1UBQ'))",
        "sys = ov.mol.prepare_system(s, water='implicit')   # fast, no box",
        "sys = ov.mol.prepare_system(s, ligand='CCO', ligand_ff='gaff-2.11')",
        "sys = ov.mol.prepare_system(docking_result)        # complex from dock()",
    ],
    related=["mol.simulate", "mol.minimize", "mol.equilibrate",
             "mol.production", "mol.dock"],
)
def prepare_system(
    structure,
    *,
    forcefield: str = "amber14-all",
    water: str = "tip3p",
    box_padding_nm: float = 1.0,
    ions: str = "0.15 M NaCl",
    ph: float = 7.0,
    keep_water: bool = False,
    ligand=None,
    ligand_ff: str = "gaff-2.11",
    hmr: bool = False,
    nonbonded_cutoff_nm: float = 1.0,
    seed: int = 0,
    verbose: bool = False,
) -> MDSystem:
    r"""Prepare a solvated, parameterised system ready for dynamics.

    Parameters
    ----------
    structure
        A :class:`~omicverse.mol.MolStructure`, a path to a PDB file, a
        PDB/gene identifier (fetched on demand), or a
        :class:`~omicverse.mol.DockingResult` — the latter rebuilds the
        receptor plus its top-ranked pose as a complex.
    forcefield
        Protein forcefield: ``'amber14-all'`` (default) or ``'charmm36'``.
    water
        Water model (``'tip3p'``, ``'tip4pew'``, ``'spce'``) or
        ``'implicit'`` for GBn2 implicit solvent — much faster and box-free,
        at the cost of explicit solvation structure.
    box_padding_nm
        Minimum distance between solute and box edge (explicit solvent only).
    ions
        Salt to add on top of charge neutralisation, e.g. ``'0.15 M NaCl'``;
        ``'none'`` neutralises only.
    ph
        pH used by PDBFixer to pick protonation states.
    keep_water
        Keep crystallographic waters.
    ligand
        Optional small molecule to include: SMILES, ``.sdf`` / ``.mol2``
        path, RDKit ``Mol``, or a ``DockingResult``.
    ligand_ff
        ``'gaff-2.11'`` or ``'openff-2.1.0'``.
    hmr
        Hydrogen-mass repartitioning — required for a 4 fs timestep.
    seed
        Seed for conformer generation / solvation randomness.

    Returns
    -------
    MDSystem

    Notes
    -----
    Explicit solvent uses PME with ``constraints=HBonds``; implicit solvent
    runs without a periodic box. Ligand parameterisation needs
    ``openmmforcefields`` + ``openff-toolkit``.
    """
    check_md()
    from openmm import app, unit

    # PDBFixer is vendored (it is not on PyPI) — see omicverse.external.pdbfixer
    from ...external.pdbfixer import PDBFixer

    ff_name = _norm_ff(forcefield)
    water_l = str(water).lower()
    implicit = water_l == "implicit"
    if not implicit and (ff_name, water_l) not in _WATER_FILES:
        raise ValueError(
            f"water model {water!r} is not available for forcefield "
            f"{forcefield!r}; supported combinations: "
            f"{sorted(k for k in _WATER_FILES if k[0] == ff_name)} "
            "or water='implicit'")

    workdir = tempfile.mkdtemp(prefix="ovmd_prep_")
    pdb_path, dock_ligand, meta = _resolve_structure(structure, workdir)
    input_hash = file_hash(pdb_path)
    if ligand is None and dock_ligand is not None:
        ligand = dock_ligand

    # ---- repair the protein -------------------------------------------
    if verbose:
        print(f"   fixing structure ({os.path.basename(pdb_path)}) at pH {ph}")
    # Everything from structure repair through solvation draws on the global
    # RNGs (see _seeded_rng), so run it all under one seeded context.
    with _seeded_rng(seed):
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=keep_water)
        fixer.findMissingAtoms()
        # PDBFixer relaxes newly built atoms with a Langevin integrator, so
        # this step is stochastic unless seeded.
        fixer.addMissingAtoms(seed=int(seed))
        fixer.addMissingHydrogens(ph)

        # ---- forcefield ------------------------------------------------
        ff_files = [_FF_FILES[ff_name]]
        ff_files.append(
            _IMPLICIT_XML if implicit else _WATER_FILES[(ff_name, water_l)])
        forcefield_obj = app.ForceField(*ff_files)

        off_mol = None
        if ligand is not None:
            if verbose:
                print(f"   parameterising ligand with {ligand_ff}")
            off_mol = _to_openff_molecule(ligand, seed=seed)
            # Charge up front rather than lazily inside createSystem: a
            # failure then points at the ligand, and a docked pose can fall
            # back to a clean conformer. openmmforcefields reuses charges
            # already present on the molecule instead of recomputing them.
            _assign_charges(off_mol, seed=seed, verbose=verbose)
            _register_ligand_template(forcefield_obj, off_mol, ligand_ff)

        # ---- assemble complex ------------------------------------------
        modeller = app.Modeller(fixer.topology, fixer.positions)
        if off_mol is not None:
            lig_top = off_mol.to_topology().to_openmm()
            lig_pos = off_mol.conformers[0].to_openmm()
            before = {r.index for r in modeller.topology.residues()}
            modeller.add(lig_top, lig_pos)
            # OpenFF names the residue 'UNK'. Rename it to the field-standard
            # 'LIG' so the obvious selection — "resname LIG" — actually works
            # in the analysis helpers; the name is also recorded in provenance
            # so callers never have to guess it.
            for res in modeller.topology.residues():
                if res.index not in before:
                    res.name = LIGAND_RESNAME

        # ---- solvate ---------------------------------------------------
        if not implicit:
            conc, pos_ion, neg_ion = _parse_ionic_strength(ions)
            if verbose:
                print(f"   solvating: {water} box, padding {box_padding_nm} nm, "
                      f"{conc} M {pos_ion}/{neg_ion}")
            modeller.addSolvent(
                forcefield_obj,
                model=_SOLVENT_MODELS.get(water_l, "tip3p"),
                padding=box_padding_nm * unit.nanometer,
                ionicStrength=conc * unit.molar,
                positiveIon=pos_ion, negativeIon=neg_ion,
            )

    # ---- build the System ---------------------------------------------
    hydrogen_mass = (4.0 * unit.amu) if hmr else None
    create_kw: Dict[str, Any] = {
        "constraints": app.HBonds,
        "rigidWater": True,
    }
    if hydrogen_mass is not None:
        create_kw["hydrogenMass"] = hydrogen_mass
    if implicit:
        create_kw["nonbondedMethod"] = app.CutoffNonPeriodic
        create_kw["nonbondedCutoff"] = 2.0 * unit.nanometer
    else:
        # PME requires cutoff <= half the shortest box edge (minimum-image
        # convention). A small solute with tight padding can violate that, so
        # shrink the cutoff rather than letting OpenMM fail deep inside
        # Context creation with an opaque error.
        box = modeller.topology.getPeriodicBoxVectors()
        edges = [box[i][i].value_in_unit(unit.nanometer) for i in range(3)]
        # 0.45 rather than 0.5: an NPT barostat shrinks the box during
        # equilibration, and OpenMM enforces cutoff < box/2 every step.
        max_cutoff = 0.45 * min(edges)
        cutoff = nonbonded_cutoff_nm
        if cutoff > max_cutoff:
            import warnings
            warnings.warn(
                f"nonbonded cutoff {cutoff:g} nm exceeds half the shortest box "
                f"edge ({min(edges):.2f} nm); reducing it to {max_cutoff:.2f} "
                "nm. Increase box_padding_nm for a physically roomier box if "
                "you need the longer cutoff.",
                RuntimeWarning, stacklevel=2)
            cutoff = max_cutoff
        create_kw["nonbondedMethod"] = app.PME
        create_kw["nonbondedCutoff"] = cutoff * unit.nanometer
        nonbonded_cutoff_nm = cutoff

    try:
        system = forcefield_obj.createSystem(modeller.topology, **create_kw)
    except Exception as exc:
        text = str(exc).lower()
        # A missing partial-charge backend is by far the most common ligand
        # failure, and its raw message ("no registered toolkits can provide
        # assign_partial_charges") says nothing about how to fix it.
        if any(k in text for k in ("assign_partial_charges", "am1bcc",
                                   "chargemethodunavailable", "toolkit")):
            raise RuntimeError(
                "ligand parameterisation failed: no backend can assign "
                "AM1-BCC partial charges.\n"
                f"  ({exc})\n"
                "Both GAFF and OpenFF need AM1-BCC charges, which require "
                "AmberTools (the `antechamber` / `sqm` programs) — conda-only:\n"
                "    conda install -c conda-forge ambertools\n"
                + _ambertools_path_hint() +
                "Alternatives:\n"
                "  * install openff-nagl for fast ML-predicted charges;\n"
                "  * or supply a ligand that is already parameterised."
            ) from exc
        raise RuntimeError(
            f"failed to build the OpenMM System: {exc}\n"
            "Common causes: a residue the forcefield does not recognise "
            "(non-standard amino acid, cofactor, metal), or a ligand that "
            "was not parameterised. Pass the cofactor via `ligand=`, or pick "
            "a forcefield that covers it.") from exc

    box_vectors = None
    if not implicit:
        box_vectors = modeller.topology.getPeriodicBoxVectors()

    n_clash = _clash_report(modeller.topology, modeller.positions)
    if n_clash > 200:
        import warnings
        warnings.warn(
            f"{n_clash} steric clashes (<0.8 A) detected after preparation. "
            "Energy minimisation may fail or the run may blow up. Check the "
            "input structure for overlapping atoms or a badly placed ligand.",
            RuntimeWarning, stacklevel=2)

    prov = base_provenance(
        stage="prepare_system",
        forcefield=ff_name, forcefield_files=ff_files, water=water_l,
        implicit_solvent=implicit, box_padding_nm=box_padding_nm,
        ions=ions, ph=ph, keep_water=keep_water, hmr=hmr,
        hydrogen_mass_amu=4.0 if hmr else None,
        nonbonded_cutoff_nm=None if implicit else nonbonded_cutoff_nm,
        constraints="HBonds", seed=seed,
        ligand_ff=ligand_ff if off_mol is not None else None,
        ligand_smiles=(off_mol.to_smiles() if off_mol is not None else None),
        ligand_resname=(LIGAND_RESNAME if off_mol is not None else None),
        ligand_selection=(f"resname {LIGAND_RESNAME}"
                          if off_mol is not None else None),
        n_atoms=int(system.getNumParticles()),
        n_clashes=n_clash,
        input_pdb=pdb_path, input_sha256=input_hash,
        structure_meta=meta,
    )

    if verbose:
        print(f"   system ready: {system.getNumParticles()} particles"
              + ("" if implicit else " (explicit solvent)"))

    return MDSystem(modeller.topology, system, modeller.positions,
                    box_vectors=box_vectors, forcefield=ff_name,
                    water=water_l, provenance=prov)
