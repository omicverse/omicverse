r"""Molecular docking for :mod:`omicverse.mol`.

:func:`dock` runs an **AutoDock Vina** docking of a small molecule into a
target structure; :func:`redock_validate` performs the mandatory
docking-protocol validation — re-docking a co-crystallized ligand into its
own receptor and checking the best-pose RMSD against the crystal pose.

Pipeline: receptor prep (PDB -> pdbqt, meeko) · ligand prep (rdkit ETKDG
3D embedding -> meeko pdbqt) · Vina search.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .._registry import register_function
from ._check import _need, check_dock

# residue names that are solvent / ions / cryo-additives — never the ligand
_NON_LIGAND = {
    "HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE",
    "CU", "NI", "CO", "CD", "SO4", "PO4", "GOL", "EDO", "PEG", "ACT",
    "DMS", "MPD", "TRS", "FMT", "NO3", "IOD", "BR", "CIT", "EPE", "BME",
}

# RCSB chemical-component descriptor endpoint
_RCSB_CHEMCOMP = "https://data.rcsb.org/rest/v1/core/chemcomp/{code}"

# the field-accepted redocking-success threshold
RMSD_PASS = 2.0


class DockingResult:
    """Output of :func:`dock` — docked poses and their Vina affinities.

    Attributes
    ----------
    poses : list of rdkit.Chem.Mol
        One RDKit molecule per pose (ranked best-first).
    pose_blocks : list of str
        Each pose as a PDB block (for :func:`omicverse.mol.view_docking`).
    affinities : numpy.ndarray
        Vina affinity per pose, kcal/mol (more negative = stronger).
    best : rdkit.Chem.Mol
        The top-ranked pose.
    ligand : rdkit.Chem.Mol
        The prepared input ligand.
    box : tuple or None
        ``(center, size)`` of the Vina search box.
    """

    def __init__(self, poses, pose_blocks, affinities, ligand, box):
        self.poses = poses
        self.pose_blocks = pose_blocks
        self.affinities = np.asarray(affinities, dtype=float)
        self.ligand = ligand
        self.box = box

    @property
    def best(self):
        return self.poses[0] if self.poses else None

    def rmsd_to(self, reference) -> float:
        """Heavy-atom RMSD of the best pose vs a reference RDKit molecule.

        Both molecules must share the same connectivity (e.g. a redocked
        ligand vs its crystal pose). Symmetry is handled by RDKit.
        """
        from rdkit.Chem import rdMolAlign
        return float(rdMolAlign.CalcRMS(self.best, reference))

    def save_poses(self, path: str, *, format: Optional[str] = None) -> str:
        r"""Write every docked pose to a file (``.sdf`` or ``.pdb``).

        ``.sdf`` preserves bond orders and tags each pose with its Vina
        affinity (``vina_affinity`` SD field) — the format MGLTools, PyMOL
        and most downstream tools expect. ``.pdb`` writes a multi-``MODEL``
        PDB; coarser but universally readable.

        Format is inferred from the extension unless ``format`` overrides.
        Returns the output path.
        """
        return self._write_poses(path, range(len(self.poses)), format)

    def save_pose(self, path: str, *, pose: int = 0,
                  format: Optional[str] = None) -> str:
        r"""Write a single docked pose to a file (``.sdf`` or ``.pdb``).

        ``pose=0`` is the top-scored pose. See :meth:`save_poses`.
        """
        if pose < 0 or pose >= len(self.poses):
            raise IndexError(
                f"pose {pose} out of range — result has "
                f"{len(self.poses)} poses")
        return self._write_poses(path, [pose], format)

    def _write_poses(self, path: str, indices, format: Optional[str]) -> str:
        from rdkit import Chem
        fmt = (format or os.path.splitext(path)[1].lstrip(".")).lower()
        if fmt in ("sdf", "sd"):
            writer = Chem.SDWriter(path)
            try:
                for i in indices:
                    mol = Chem.Mol(self.poses[i])
                    mol.SetProp("_Name", f"pose_{i + 1}")
                    mol.SetProp("vina_affinity_kcal_mol",
                                f"{float(self.affinities[i]):.3f}")
                    mol.SetProp("pose_rank", str(i + 1))
                    writer.write(mol)
            finally:
                writer.close()
            return path
        if fmt == "pdb":
            with open(path, "w") as fh:
                for n, i in enumerate(indices, start=1):
                    fh.write(f"MODEL     {n}\n")
                    for line in self.pose_blocks[i].splitlines():
                        if line.startswith(("MODEL", "ENDMDL", "END ", "END\n")):
                            continue
                        fh.write(line + "\n")
                    fh.write(f"REMARK   1 VINA_AFFINITY "
                             f"{float(self.affinities[i]):.3f}\n")
                    fh.write("ENDMDL\n")
                fh.write("END\n")
            return path
        raise ValueError(
            f"unsupported format {fmt!r}; use '.sdf' (recommended) or '.pdb'")

    def __repr__(self) -> str:
        if not len(self.affinities):
            return "DockingResult(no poses)"
        return (f"DockingResult({len(self.poses)} poses, "
                f"best {self.affinities[0]:.2f} kcal/mol)")


# ------------------------------------------------------------------ #
# ligand / receptor preparation                                       #
# ------------------------------------------------------------------ #


_LIGAND_FILE_EXTS = (".sdf", ".mol", ".mol2", ".pdb", ".smi", ".smiles")


def _load_ligand_file(path: str):
    """Load an RDKit Mol from a local ligand file — sdf/mol/mol2/pdb/smi."""
    from rdkit import Chem

    ext = os.path.splitext(path)[1].lower()
    if ext == ".sdf":
        for m in Chem.SDMolSupplier(path, removeHs=False):
            if m is not None:
                return m
        raise ValueError(f"no parseable molecule in SDF: {path}")
    if ext == ".mol":
        m = Chem.MolFromMolFile(path, removeHs=False)
    elif ext == ".mol2":
        m = Chem.MolFromMol2File(path, removeHs=False)
    elif ext == ".pdb":
        m = Chem.MolFromPDBFile(path, removeHs=False)
    elif ext in (".smi", ".smiles"):
        with open(path) as fh:
            first = fh.readline().strip()
        if not first:
            raise ValueError(f"empty SMILES file: {path}")
        # SMILES files can have "<smiles> <name>" — take the first token
        m = Chem.MolFromSmiles(first.split()[0])
    else:
        raise ValueError(
            f"unsupported ligand file extension {ext!r} — "
            f"accepted: {', '.join(_LIGAND_FILE_EXTS)}")
    if m is None:
        raise ValueError(f"RDKit failed to parse ligand file: {path}")
    return m


def _resolve_ligand(ligand: Any):
    """Resolve a ligand spec to an RDKit Mol.

    Accepts an RDKit ``Mol``, a local file path
    (``.sdf`` / ``.mol`` / ``.mol2`` / ``.pdb`` / ``.smi``), a SMILES
    string, a drug name (resolved via ChEMBL), or a row / dict carrying
    a ``smiles`` key (e.g. one row of ``known_drugs()``).
    """
    from rdkit import Chem

    if isinstance(ligand, Chem.Mol):
        return ligand
    # a known_drugs() row / dict
    smiles = None
    if hasattr(ligand, "get") and not isinstance(ligand, str):
        smiles = ligand.get("smiles")
    elif hasattr(ligand, "__getitem__") and not isinstance(ligand, str):
        try:
            smiles = ligand["smiles"]
        except Exception:
            smiles = None
    if smiles:
        return Chem.MolFromSmiles(smiles)

    if isinstance(ligand, str):
        # local file path with a known molecular-file extension
        ext = os.path.splitext(ligand)[1].lower()
        if ext in _LIGAND_FILE_EXTS and os.path.exists(ligand):
            return _load_ligand_file(ligand)
        # probe as SMILES — silence RDKit's parser log, since a failure
        # here just means the string is a drug name, not an error
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        try:
            mol = Chem.MolFromSmiles(ligand)
        finally:
            RDLogger.EnableLog("rdApp.*")
        if mol is not None:
            return mol
        # treat as a drug name -> ChEMBL
        _need("chembl_webresource_client", "mol", "ov.mol ligand-name lookup")
        from chembl_webresource_client.new_client import new_client
        hits = list(new_client.molecule.filter(
            pref_name__iexact=ligand).only(["molecule_structures"]))
        if hits and hits[0].get("molecule_structures"):
            smi = hits[0]["molecule_structures"].get("canonical_smiles")
            if smi:
                return Chem.MolFromSmiles(smi)
        raise ValueError(f"could not resolve ligand {ligand!r} "
                          "(not a valid SMILES, no ChEMBL drug by that name, "
                          "no such ligand file)")
    raise TypeError(f"unsupported ligand type: {type(ligand)}")


def _prepare_ligand_pdbqt(mol, seed: int) -> Tuple[str, Any]:
    """3D-embed an RDKit Mol (ETKDG) and write a Vina-ready pdbqt string."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError("3D embedding (ETKDG) failed for the ligand")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:  # pragma: no cover - geometry fallback
        pass

    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(f"meeko ligand pdbqt conversion failed: {err}")
    return pdbqt, mol


def _find_meeko_script(name: str) -> Optional[str]:
    """Locate a meeko console script — on PATH or beside the interpreter."""
    exe = shutil.which(name)
    if exe:
        return exe
    import sys
    cand = os.path.join(os.path.dirname(sys.executable), name)
    return cand if os.path.exists(cand) else None


def _prepare_receptor_pdbqt(structure, workdir: str) -> str:
    """Convert a protein structure to a Vina-ready receptor pdbqt.

    Real PDB structures routinely carry residues with missing side-chain
    atoms or no matching template; ``-a`` deletes them, and any
    template-failure residues named in the error are deleted on a retry —
    so docking does not abort on a normal experimental structure.
    """
    import re
    import sys
    exe = _find_meeko_script("mk_prepare_receptor.py")
    if exe is None:
        raise ImportError(
            "ov.mol docking needs meeko's 'mk_prepare_receptor.py' — "
            "install the docking extra: pip install 'omicverse[mol-dock]'")
    rec_pdb = os.path.join(workdir, "receptor.pdb")
    structure.to_pdb(rec_pdb)
    base = os.path.join(workdir, "receptor")
    candidates = (base + ".pdbqt", base + "_rigid.pdbqt")

    def _run(extra):
        for c in candidates:
            if os.path.exists(c):
                os.remove(c)
        cmd = [sys.executable, exe, "--read_pdb", rec_pdb, "-o", base,
               "-p", "-a", "--default_altloc", "A"] + extra
        proc = subprocess.run(cmd, capture_output=True, text=True)
        for c in candidates:
            if os.path.exists(c):
                return c, proc
        return None, proc

    out, proc = _run([])
    if out is None:
        # delete the residues meeko named as template failures, then retry
        bad = re.findall(r"[A-Za-z0-9]+:\d+",
                         proc.stdout + " " + proc.stderr)
        if bad:
            out, proc = _run(["-d", ",".join(sorted(set(bad)))])
    if out is not None:
        return out
    raise RuntimeError(
        "receptor preparation produced no pdbqt.\n"
        f"  stdout: {proc.stdout[-700:]}\n  stderr: {proc.stderr[-400:]}")


# ------------------------------------------------------------------ #
# search box                                                          #
# ------------------------------------------------------------------ #


def _box_from_coords(coords: np.ndarray, padding: float = 8.0,
                     minimum: float = 16.0) -> Tuple[list, list]:
    """A center / size box enclosing a coordinate set, plus padding.

    The box is centred on the **centroid** of the points, not the
    bounding-box midpoint — for an asymmetric ligand (e.g. a rigid core
    with a flexible tail) those differ by several Angstrom, and a
    midpoint-centred box shifts the true binding mode toward the box edge,
    which makes Vina rank spurious poses first.
    """
    center = coords.mean(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    size = np.maximum(extent + 2.0 * padding, minimum)
    return center.tolist(), size.tolist()


def _resolve_box(structure, pocket, box) -> Tuple[list, list]:
    """Resolve the Vina search box from a pocket id / explicit box / blind."""
    if box is not None:
        center, size = box
        return list(center), list(size)
    if pocket is not None:
        spheres = (structure.meta or {}).get("_pocket_spheres", {})
        coords = spheres.get(int(pocket))
        if not coords:
            raise ValueError(
                f"pocket {pocket} has no alpha-spheres — run "
                f"ov.mol.pockets(structure) before docking into a pocket.")
        return _box_from_coords(np.asarray(coords, dtype=float),
                                padding=5.0, minimum=18.0)
    # blind docking — the whole protein
    return _box_from_coords(np.asarray(structure.atoms.coord, dtype=float))


# ------------------------------------------------------------------ #
# Vina runner                                                         #
# ------------------------------------------------------------------ #


def _run_vina(receptor_pdbqt: str, ligand_pdbqt: str, center, size,
              exhaustiveness: int, n_poses: int, seed: int, verbose: bool):
    """Run a Vina docking and return (pose_mols, pose_blocks, affinities)."""
    from vina import Vina
    from rdkit import Chem
    from meeko import PDBQTMolecule, RDKitMolCreate

    v = Vina(sf_name="vina", seed=int(seed), verbosity=1 if verbose else 0)
    v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt)
    v.set_ligand_from_string(ligand_pdbqt)
    v.compute_vina_maps(center=[float(c) for c in center],
                        box_size=[float(s) for s in size])
    v.dock(exhaustiveness=int(exhaustiveness), n_poses=int(n_poses))

    energies = np.asarray(v.energies(n_poses=n_poses), dtype=float)
    affinities = energies[:, 0] if energies.ndim == 2 else energies

    pdbqt_mol = PDBQTMolecule(v.poses(n_poses=n_poses), skip_typing=True)
    mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    ligand_mol = mols[0]

    poses, blocks = [], []
    for conf_id in range(ligand_mol.GetNumConformers()):
        pose = Chem.Mol(ligand_mol)
        pose.RemoveAllConformers()
        pose.AddConformer(ligand_mol.GetConformer(conf_id), assignId=True)
        poses.append(pose)
        blocks.append(Chem.MolToPDBBlock(pose))
    return poses, blocks, affinities[:len(poses)]


# ------------------------------------------------------------------ #
# public                                                              #
# ------------------------------------------------------------------ #


@register_function(
    aliases=["dock", "分子对接", "molecular_docking", "vina", "docking",
             "autodock", "对接"],
    category="mol",
    description=(
        "Dock a small molecule into a target protein structure with "
        "AutoDock Vina. The ligand accepts five inputs: a SMILES string, "
        "a drug name (resolved via ChEMBL), an RDKit Mol, a known_drugs() "
        "row, or a local file path (.sdf / .mol / .mol2 / .pdb / .smi) — "
        "use the file form to dock compounds from a vendor catalog, an "
        "in-house design pipeline, or a virtual-screening hit list. The "
        "search box is taken from a detected pocket, given explicitly, or "
        "spans the whole protein (blind docking). Returns a DockingResult "
        "with ranked poses, Vina affinities, and save_poses() / "
        "save_pose() helpers to write the poses to SDF or PDB. Docking is "
        "stochastic — results are reproducible for a fixed seed."
    ),
    examples=[
        "result = ov.mol.dock(s, 'gefitinib', pocket=1)",
        "result = ov.mol.dock(s, 'CCOc1cc2ncnc(Nc3ccc...)c2cc1', pocket=1)",
        "result = ov.mol.dock(s, '/path/to/my_compound.sdf', pocket=1)",
        "result = ov.mol.dock(s, known_drugs('EGFR').iloc[0], pocket=1)",
        "result.save_poses('poses.sdf')",
    ],
    related=["mol.redock_validate", "mol.view_docking", "mol.pockets"],
)
def dock(structure, ligand: Any, *, pocket: Optional[int] = None,
         box: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
         exhaustiveness: int = 8, n_poses: int = 9, seed: int = 0,
         verbose: bool = False) -> DockingResult:
    r"""Dock a small molecule into a target structure.

    Parameters
    ----------
    structure : MolStructure
        The receptor.
    ligand
        Accepts five forms:

        - a **SMILES string** (e.g. ``'c1ccccc1'``);
        - a **drug name** (resolved via ChEMBL — e.g. ``'gefitinib'``);
        - an **RDKit** :class:`~rdkit.Chem.Mol`;
        - a row / dict carrying a ``smiles`` key (e.g. a row of
          :func:`omicverse.mol.known_drugs`);
        - a **local file path** with a molecular-file extension —
          ``.sdf`` / ``.mol`` / ``.mol2`` / ``.pdb`` / ``.smi``. Use this
          to dock a compound from a vendor SDF, an in-house design
          pipeline, or a virtual-screening hit list.
    pocket
        A ``pocket_id`` from :func:`omicverse.mol.pockets` — its
        alpha-spheres define the search box.
    box
        Explicit ``(center, size)`` search box, overriding ``pocket``.
    exhaustiveness
        Vina search exhaustiveness (higher = more thorough, slower).
    n_poses
        Number of poses to return.
    seed
        Vina random seed. Docking is stochastic — pass a fixed non-zero
        seed for reproducible poses; ``seed=0`` lets Vina pick a random
        seed each run (the Vina convention).
    verbose
        Print Vina progress.

    Returns
    -------
    DockingResult
    """
    check_dock()
    mol = _resolve_ligand(ligand)
    if mol is None:
        raise ValueError("could not build an RDKit molecule from `ligand`")
    center, size = _resolve_box(structure, pocket, box)

    with tempfile.TemporaryDirectory() as wd:
        receptor_pdbqt = _prepare_receptor_pdbqt(structure, wd)
        ligand_pdbqt, prepared = _prepare_ligand_pdbqt(mol, seed)
        if verbose:
            print(f"docking into box center={[round(c, 1) for c in center]} "
                  f"size={[round(s, 1) for s in size]}")
        poses, blocks, affinities = _run_vina(
            receptor_pdbqt, ligand_pdbqt, center, size,
            exhaustiveness, n_poses, seed, verbose)

    return DockingResult(poses, blocks, affinities, prepared,
                         (center, size))


def _extract_cocrystal_ligand(pdb_path: str):
    """Pull the largest non-solvent HETATM residue out of an experimental PDB.

    Returns ``(resname, rdkit_mol_with_crystal_coords)`` or ``None``.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    groups: dict = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in _NON_LIGAND:
                continue
            key = (resname, line[21], line[22:26].strip())
            groups.setdefault(key, []).append(line)
    if not groups:
        return None
    (resname, _, _), lines = max(groups.items(), key=lambda kv: len(kv[1]))
    if len(lines) < 6:  # too small to be a drug-like ligand
        return None

    block = "".join(lines) + "END\n"
    crystal = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    if crystal is None:
        return None

    # assign correct bond orders from the RCSB chemical-component SMILES
    import requests
    try:
        meta = requests.get(_RCSB_CHEMCOMP.format(code=resname),
                            timeout=60).json()
        desc = meta.get("rcsb_chem_comp_descriptor", {})
        smiles = desc.get("SMILES_stereo") or desc.get("SMILES")
        template = Chem.MolFromSmiles(smiles) if smiles else None
        if template is not None:
            crystal = AllChem.AssignBondOrdersFromTemplate(template, crystal)
            return resname, crystal, smiles
    except Exception:  # pragma: no cover - best-effort enrichment
        pass
    return resname, crystal, None


@register_function(
    aliases=["redock_validate", "重对接验证", "redock", "docking_validation",
             "validate_docking", "redocking"],
    category="mol",
    description=(
        "Validate a docking protocol by redocking — re-dock a "
        "co-crystallized reference ligand into its own experimental "
        "receptor and measure the best-pose RMSD against the crystal "
        "pose. A protocol is trusted only when it reproduces the native "
        "binding mode within the field-accepted RMSD < 2.0 A threshold. "
        "Run this before trusting any dock() result on a new ligand."
    ),
    examples=[
        "val = ov.mol.redock_validate(experimental_structure)",
        "val['rmsd'], val['passed']",
    ],
    related=["mol.dock", "mol.view_docking"],
)
def redock_validate(structure, ref_ligand: Any = None, *,
                    exhaustiveness: int = 8, seed: int = 0,
                    verbose: bool = False) -> dict:
    r"""Redocking validation of the docking protocol.

    Parameters
    ----------
    structure : MolStructure
        An experimental structure containing a co-crystallized ligand.
    ref_ligand
        Reference ligand. ``None`` auto-extracts the co-crystal ligand
        from ``structure``.
    exhaustiveness, seed
        Passed through to Vina.
    verbose
        Print progress.

    Returns
    -------
    dict
        ``rmsd`` — top-scored pose RMSD vs the crystal pose (Angstrom) —
        the headline redocking metric;
        ``passed`` — ``rmsd < 2.0``: the strict criterion, validating
        *search + scoring* together;
        ``min_rmsd`` — best RMSD across *all* poses;
        ``min_rmsd_rank`` — 1-based rank of that pose;
        ``sampling_passed`` — ``min_rmsd < 2.0``: the search reproduced
        the native binding mode (it may rank below pose 1 — Vina's
        semi-empirical score does not always rank the native pose first);
        ``all_rmsd`` — per-pose RMSD list;
        ``affinity`` — top-scored pose Vina affinity;
        ``ref_ligand`` — the reference ligand's residue name;
        ``result`` — the underlying :class:`DockingResult`.
    """
    check_dock()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    extracted = None
    if ref_ligand is None:
        extracted = _extract_cocrystal_ligand(structure.path)
        if extracted is None:
            raise ValueError(
                "no co-crystallized ligand found in the structure — pass "
                "`ref_ligand` explicitly, or use an experimental PDB that "
                "contains a bound ligand.")
        resname, crystal, smiles = extracted
        ligand = smiles or Chem.MolToSmiles(Chem.RemoveHs(crystal))
    else:
        resname = "REF"
        crystal = None
        ligand = ref_ligand

    # the search box is centred on the crystal ligand — kept snug
    # (ligand extent + ~4 A per side): an over-large box lets Vina drift
    # into spurious surface poses and rank them above the native one
    if extracted is not None:
        conf = crystal.GetConformer()
        coords = np.array([list(conf.GetAtomPosition(i))
                           for i in range(crystal.GetNumAtoms())])
        box = _box_from_coords(coords, padding=4.0, minimum=14.0)
    else:
        box = None

    result = dock(structure, ligand, box=box, pocket=None,
                  exhaustiveness=exhaustiveness, n_poses=9, seed=seed,
                  verbose=verbose)

    # RMSD of every pose vs the crystal coordinates (no realignment —
    # both are already in the receptor frame); symmetry handled by RDKit
    all_rmsd: list = []
    if crystal is not None:
        from rdkit.Chem import rdMolAlign
        ref = Chem.RemoveHs(crystal)
        for pose in result.poses:
            try:
                probe = AllChem.AssignBondOrdersFromTemplate(
                    ref, Chem.RemoveHs(pose))
                all_rmsd.append(float(rdMolAlign.CalcRMS(probe, ref)))
            except Exception as exc:  # pragma: no cover
                if verbose:
                    print(f"RMSD computation fell back for a pose: {exc}")
                all_rmsd.append(float("nan"))

    rmsd = all_rmsd[0] if all_rmsd else float("nan")
    finite = [(r, i) for i, r in enumerate(all_rmsd) if r == r]
    min_rmsd, min_idx = min(finite) if finite else (float("nan"), -1)

    return {
        "rmsd": rmsd,
        "passed": bool(rmsd < RMSD_PASS) if rmsd == rmsd else False,
        "min_rmsd": min_rmsd,
        "min_rmsd_rank": min_idx + 1 if min_idx >= 0 else None,
        "sampling_passed": bool(min_rmsd < RMSD_PASS)
        if min_rmsd == min_rmsd else False,
        "all_rmsd": all_rmsd,
        "affinity": float(result.affinities[0]) if len(result.affinities)
        else float("nan"),
        "ref_ligand": resname,
        "result": result,
    }
