"""Acceptance tests for ``ov.mol``'s pharmacophore alignment layer.

Tiers
-----
1. **Always** — the public names exist, the lazy-import contract holds, and
   calling one without RDKit raises an actionable ImportError.
2. **rdkit installed** — the geometry is checked against models built *from a
   molecule's own conformer*, so the expected answer is known without any
   stored constant: features placed on a real conformer must be reachable, a
   pose must stay out of excluded volumes, and a model no ligand feature can
   satisfy must be reported rather than silently scored.

Everything here is synthetic: models are derived at test time from RDKit
conformers of small SMILES, never from a dataset.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None
rdkit_only = pytest.mark.skipif(not _HAS_RDKIT, reason="rdkit not installed")

PUBLIC = ("pharmacophore_features", "pharmacophore_score", "pose_strain_energy",
          "pharmacophore_align", "align_pharmacophore_set", "PharmacophorePose")


def test_public_surface_is_exposed():
    import omicverse.mol as mol

    for name in PUBLIC:
        assert name in dir(mol)
        assert getattr(mol, name) is not None


def test_registry_entries_exist():
    from omicverse._registry import get_registry
    from omicverse.mol import _pharmacophore  # noqa: F401  (populates the registry)

    reg = get_registry()
    for name in ("pharmacophore_align", "pharmacophore_score",
                 "pose_strain_energy", "align_pharmacophore_set",
                 "pharmacophore_features"):
        assert reg.get_function(name) is not None, f"{name} is not registered"
    # discoverable from the intent, not only from the exact name
    assert reg.find("pharmacophore"), "no registry hit for 'pharmacophore'"


@pytest.mark.skipif(_HAS_RDKIT, reason="only meaningful without rdkit")
def test_missing_rdkit_is_actionable():
    import omicverse.mol as mol

    with pytest.raises(ImportError, match="rdkit"):
        mol.pharmacophore_align("CCO", [{"family": "Acceptor", "center": [0, 0, 0],
                                         "radius": 1.0}])


# --------------------------------------------------------------------------- #
# helpers (rdkit tier)                                                        #
# --------------------------------------------------------------------------- #
def _conformer(smiles: str, seed: int = 11):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s")
    return mol


def _model_from_conformer(smiles: str, families=("Acceptor", "Donor"), radius=0.9):
    """A pharmacophore model that the molecule itself satisfies exactly.

    Taking the spheres from a real conformer's own feature points is what makes
    the expected answer known: a correct alignment can reach every sphere.
    """
    import omicverse.mol as mol_ns

    mol = _conformer(smiles)
    points = mol_ns.pharmacophore_features(mol)
    volumes = []
    for family in families:
        for point in points[family]:
            volumes.append({"family": family, "center": [float(c) for c in point],
                            "radius": radius, "weight": 1.0})
    return mol, volumes


# --------------------------------------------------------------------------- #
# features and scoring                                                        #
# --------------------------------------------------------------------------- #
@rdkit_only
def test_features_are_per_family_points():
    import omicverse.mol as mol_ns

    mol = _conformer("c1ccccc1O")
    points = mol_ns.pharmacophore_features(mol)
    assert set(points) == {"Donor", "Acceptor", "Hydrophobe", "Aromatic"}
    assert points["Aromatic"].shape[1] == 3
    assert points["Aromatic"].shape[0] >= 1
    assert points["Donor"].shape[0] >= 1
    # the aromatic feature is the ring centroid, i.e. inside the molecule's hull
    coords = np.asarray(mol.GetConformer().GetPositions())
    centroid = points["Aromatic"][0]
    assert np.all(centroid >= coords.min(axis=0) - 1e-6)
    assert np.all(centroid <= coords.max(axis=0) + 1e-6)


@rdkit_only
def test_hydrophobe_fallback_only_when_empty():
    import omicverse.mol as mol_ns

    mol = _conformer("CCCC")
    with_fallback = mol_ns.pharmacophore_features(mol, hydrophobe_fallback=True)
    without = mol_ns.pharmacophore_features(mol, hydrophobe_fallback=False)
    assert with_fallback["Hydrophobe"].shape[0] >= without["Hydrophobe"].shape[0]


@rdkit_only
def test_score_is_maximal_on_the_pose_the_model_came_from():
    import omicverse.mol as mol_ns

    mol, volumes = _model_from_conformer("CC(=O)NCCO")
    score = mol_ns.pharmacophore_score(mol, volumes, sigma=1.0)
    assert score == pytest.approx(sum(v["weight"] for v in volumes), rel=1e-6)


@rdkit_only
def test_score_decays_with_distance_and_respects_weights():
    import omicverse.mol as mol_ns

    mol, volumes = _model_from_conformer("CC(=O)NCCO")
    moved = [dict(v, center=[v["center"][0] + 5.0, v["center"][1], v["center"][2]])
             for v in volumes]
    assert mol_ns.pharmacophore_score(mol, moved, sigma=1.0) < \
        mol_ns.pharmacophore_score(mol, volumes, sigma=1.0)
    doubled = [dict(v, weight=2.0) for v in volumes]
    assert mol_ns.pharmacophore_score(mol, doubled, sigma=1.0) == pytest.approx(
        2 * mol_ns.pharmacophore_score(mol, volumes, sigma=1.0), rel=1e-6)
    total, contrib = mol_ns.pharmacophore_score(mol, volumes, sigma=1.0, per_volume=True)
    assert contrib.shape == (len(volumes),)
    assert total == pytest.approx(float(contrib.sum()), rel=1e-9)


@rdkit_only
def test_pose_strain_energy_rises_when_an_atom_is_displaced():
    from rdkit import Chem

    import omicverse.mol as mol_ns

    mol = Chem.RemoveHs(_conformer("CC(=O)NCCO"))
    relaxed = mol_ns.pose_strain_energy(mol)
    distorted = Chem.Mol(mol)
    pos = distorted.GetConformer().GetAtomPosition(0)
    distorted.GetConformer().SetAtomPosition(0, (pos.x + 2.0, pos.y, pos.z))
    assert mol_ns.pose_strain_energy(distorted) > relaxed


# --------------------------------------------------------------------------- #
# alignment                                                                   #
# --------------------------------------------------------------------------- #
@rdkit_only
def test_align_reaches_a_model_the_ligand_can_satisfy():
    import omicverse.mol as mol_ns

    smiles = "CC(=O)NCCO"
    _mol, volumes = _model_from_conformer(smiles)
    pose = mol_ns.pharmacophore_align(smiles, volumes, sigma=1.0, n_confs=40,
                                      seed=3, n_threads=1, refine_iters=40)
    assert isinstance(pose, mol_ns.PharmacophorePose)
    assert pose.max_score == pytest.approx(len(volumes))
    assert pose.score_fraction > 0.6
    assert pose.coords.shape == (pose.mol.GetNumAtoms(), 3)
    assert np.isfinite(pose.coords).all()
    assert set(pose.to_dict()) >= {"score", "max_score", "score_fraction",
                                   "clash_depth", "conformer_energy", "strain"}


@rdkit_only
def test_align_keeps_topology_and_hydrogen_policy():
    from rdkit import Chem

    import omicverse.mol as mol_ns

    smiles = "C[C@H](O)C(=O)NC"
    _mol, volumes = _model_from_conformer(smiles)
    pose = mol_ns.pharmacophore_align(smiles, volumes, sigma=1.0, n_confs=30,
                                      seed=5, n_threads=1)
    assert Chem.MolToSmiles(Chem.RemoveHs(pose.mol), isomericSmiles=True) == \
        Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
    assert not any(a.GetAtomicNum() == 1 for a in pose.mol.GetAtoms())
    with_hs = mol_ns.pharmacophore_align(smiles, volumes, sigma=1.0, n_confs=10,
                                         seed=5, n_threads=1, keep_hydrogens=True)
    assert any(a.GetAtomicNum() == 1 for a in with_hs.mol.GetAtoms())


@rdkit_only
def test_align_stays_out_of_excluded_volumes():
    import omicverse.mol as mol_ns

    smiles = "CC(=O)NCCO"
    mol, volumes = _model_from_conformer(smiles)
    # Spheres placed on the reference conformer's own atoms: the pose that
    # matches the model perfectly is now forbidden, so a correct search must
    # find a different placement rather than return a clashing one.
    coords = np.asarray(mol.GetConformer().GetPositions())
    excluded = [{"center": [float(c) for c in coords[i]], "radius": 1.5}
                for i in range(0, coords.shape[0], 3)]
    pose = mol_ns.pharmacophore_align(smiles, volumes, excluded_volumes=excluded,
                                      sigma=1.0, n_confs=40, seed=7, n_threads=1)
    centres = np.asarray([s["center"] for s in excluded])
    radii = np.asarray([s["radius"] for s in excluded])
    dist = np.linalg.norm(pose.coords[:, None, :] - centres[None, :, :], axis=2)
    assert pose.clash_depth == pytest.approx(float(np.max(np.maximum(radii - dist.min(axis=0), 0.0))),
                                             abs=1e-6)
    assert pose.clash_depth <= 1e-6


@rdkit_only
def test_align_is_deterministic_for_a_fixed_seed():
    import omicverse.mol as mol_ns

    smiles = "CC(=O)NCCO"
    _mol, volumes = _model_from_conformer(smiles)
    kwargs = dict(sigma=1.0, n_confs=20, seed=13, n_threads=1, refine_iters=20)
    first = mol_ns.pharmacophore_align(smiles, volumes, **kwargs)
    second = mol_ns.pharmacophore_align(smiles, volumes, **kwargs)
    assert first.score == pytest.approx(second.score)
    np.testing.assert_allclose(first.coords, second.coords, atol=1e-8)


@rdkit_only
def test_align_rejects_a_model_with_no_matching_feature():
    import omicverse.mol as mol_ns

    with pytest.raises(RuntimeError):
        mol_ns.pharmacophore_align(
            "CCCC", [{"family": "Donor", "center": [0.0, 0.0, 0.0], "radius": 1.0}],
            sigma=1.0, n_confs=5, seed=1, n_threads=1)
    with pytest.raises(ValueError):
        mol_ns.pharmacophore_align("CCCC", [], n_confs=5)


@rdkit_only
def test_min_score_fraction_trades_overlap_for_relaxation():
    import omicverse.mol as mol_ns

    smiles = "CCCCOC(=O)CCN"
    _mol, volumes = _model_from_conformer(smiles)
    kwargs = dict(sigma=1.0, n_confs=60, seed=17, n_threads=1, refine_iters=40)
    greedy = mol_ns.pharmacophore_align(smiles, volumes, **kwargs)
    modest = mol_ns.pharmacophore_align(smiles, volumes, min_score_fraction=0.5, **kwargs)
    assert modest.score <= greedy.score + 1e-9
    assert modest.conformer_energy <= greedy.conformer_energy + 1e-6


# --------------------------------------------------------------------------- #
# batch                                                                       #
# --------------------------------------------------------------------------- #
@rdkit_only
def test_align_set_writes_one_record_per_entry_in_order(tmp_path):
    from rdkit import Chem

    import omicverse.mol as mol_ns

    specs = {}
    for smiles in ("CC(=O)NCCO", "c1ccccc1O", "CCCCO"):
        _mol, volumes = _model_from_conformer(smiles)
        specs[f"entry::{smiles}"] = {"smiles": smiles, "interaction_volumes": volumes}
    out = tmp_path / "poses.sdf"
    records = mol_ns.align_pharmacophore_set(specs, output=str(out), id_prop="ENTRY_ID",
                                             sigma=1.0, n_confs=20, seed=2, n_threads=1,
                                             refine_iters=20)
    assert [r["id"] for r in records] == list(specs)
    read = [m for m in Chem.SDMolSupplier(str(out), removeHs=False) if m is not None]
    assert len(read) == len(specs)
    for mol, (ident, spec) in zip(read, specs.items()):
        assert mol.GetProp("ENTRY_ID") == ident
        assert Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True) == \
            Chem.MolToSmiles(Chem.MolFromSmiles(spec["smiles"]), isomericSmiles=True)
        assert mol.HasProp("PH4_SCORE")


@rdkit_only
def test_align_set_fails_soft_and_keeps_the_record_count(tmp_path):
    from rdkit import Chem

    import omicverse.mol as mol_ns

    good_smiles = "CC(=O)NCCO"
    _mol, volumes = _model_from_conformer(good_smiles)
    specs = [
        {"id": "ok", "smiles": good_smiles, "interaction_volumes": volumes},
        # no ligand feature of this family exists -> the entry cannot be solved
        {"id": "impossible", "smiles": "CCCC",
         "interaction_volumes": [{"family": "Donor", "center": [0.0, 0.0, 0.0],
                                  "radius": 1.0}]},
    ]
    out = tmp_path / "poses.sdf"
    records = mol_ns.align_pharmacophore_set(specs, output=str(out), id_prop="ENTRY_ID",
                                             sigma=1.0, n_confs=10, seed=4, n_threads=1)
    assert [r["id"] for r in records] == ["ok", "impossible"]
    assert records[0]["ok"] is True
    assert records[1]["ok"] is False and "error" in records[1]
    read = [m for m in Chem.SDMolSupplier(str(out), removeHs=False) if m is not None]
    assert len(read) == 2
    assert read[1].GetProp("ENTRY_ID") == "impossible"
