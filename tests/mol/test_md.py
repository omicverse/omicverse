"""Acceptance tests for the ov.mol molecular-dynamics layer.

Tiers
-----
1. **Always** — ``import omicverse`` / ``import omicverse.mol`` must work with
   no MD backend installed, the public names must exist, and calling one must
   raise an actionable ImportError. This is the regression guard for the
   lazy-import contract.
2. **openmm installed** — CPU smoke test: prepare (implicit solvent) →
   minimize → production, checking the trajectory is real and readable.
3. **CUDA available** — full GPU pipeline, protein-ligand + MM-GBSA,
   analysis, and checkpoint restart.

GPU tiers *skip loudly* rather than silently passing when no GPU is present.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_HAS_OPENMM = importlib.util.find_spec("openmm") is not None
_HAS_MDTRAJ = importlib.util.find_spec("mdtraj") is not None
def _ligand_ff_available() -> bool:
    """True only if a ligand can actually be parameterised end to end.

    Having openmmforcefields + openff-toolkit importable is not enough: GAFF
    and OpenFF both need AM1-BCC charges from AmberTools, which the OpenFF
    toolkit locates on PATH. Check the capability, not the import.
    """
    try:
        import openmmforcefields  # noqa: F401
        from openff.toolkit.utils.toolkits import AmberToolsToolkitWrapper
        return bool(AmberToolsToolkitWrapper.is_available())
    except Exception:
        return False


_HAS_LIGAND_FF = _ligand_ff_available()


def _cuda_available() -> bool:
    if not _HAS_OPENMM:
        return False
    try:
        import openmm
        openmm.Platform.getPlatformByName("CUDA")
        return True
    except Exception:
        return False


_HAS_CUDA = _cuda_available()

needs_openmm = pytest.mark.skipif(
    not _HAS_OPENMM,
    reason="openmm not installed — install omicverse[md] to run MD tests")
needs_mdtraj = pytest.mark.skipif(
    not _HAS_MDTRAJ, reason="mdtraj not installed (omicverse[md])")
needs_gpu = pytest.mark.skipif(
    not _HAS_CUDA,
    reason="no CUDA platform in this OpenMM build — GPU tests SKIPPED (not passed)")
needs_ligand_ff = pytest.mark.skipif(
    not _HAS_LIGAND_FF,
    reason=("ligand parameterisation unavailable — needs openmmforcefields + "
            "openff-toolkit + AmberTools on PATH "
            "(conda install -c conda-forge openmmforcefields openff-toolkit "
            "ambertools)"))
needs_network = pytest.mark.skipif(
    not os.environ.get("OMICOS_MD_NETWORK_TESTS"),
    reason="needs network (RCSB) — set OMICOS_MD_NETWORK_TESTS=1")


# Tri-alanine in an extended (beta) conformation, heavy atoms only — PDBFixer
# adds the hydrogens and terminal atoms. Built with ideal internal coordinates
# (N-CA 1.458, CA-C 1.525, peptide C-N 1.329 A), so OpenMM perceives the
# peptide bonds and matches standard residue templates. The smallest system
# that exercises the whole pipeline on CPU in seconds.
#
# NB: ACE/NME caps are deliberately avoided — PDBFixer's removeHeterogens()
# strips them, which would leave an unmatched bare residue.
TRI_ALANINE_PDB = """\
ATOM      1 N    ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2 CA   ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3 CB   ALA A   1       1.991  -0.765   1.202  1.00  0.00           C
ATOM      4 C    ALA A   1       2.009   1.422   0.000  1.00  0.00           C
ATOM      5 O    ALA A   1       1.540   2.280  -0.748  1.00  0.00           O
ATOM      6 N    ALA A   2       3.008   1.664   0.843  1.00  0.00           N
ATOM      7 CA   ALA A   2       3.625   2.981   0.942  1.00  0.00           C
ATOM      8 CB   ALA A   2       3.268   3.829  -0.269  1.00  0.00           C
ATOM      9 C    ALA A   2       5.140   2.872   1.075  1.00  0.00           C
ATOM     10 O    ALA A   2       5.648   2.057   1.845  1.00  0.00           O
ATOM     11 N    ALA A   3       5.856   3.698   0.320  1.00  0.00           N
ATOM     12 CA   ALA A   3       7.314   3.696   0.352  1.00  0.00           C
ATOM     13 CB   ALA A   3       7.818   2.972   1.591  1.00  0.00           C
ATOM     14 C    ALA A   3       7.869   5.116   0.315  1.00  0.00           C
ATOM     15 O    ALA A   3       7.418   5.948  -0.472  1.00  0.00           O
TER
END
"""


@pytest.fixture
def dipeptide_pdb(tmp_path):
    """Write the tri-alanine test system to a temporary PDB."""
    path = tmp_path / "tri_ala.pdb"
    path.write_text(TRI_ALANINE_PDB)
    return str(path)


# --------------------------------------------------------------------- #
# 1. lazy-import contract — runs everywhere, including without [md]
# --------------------------------------------------------------------- #
class TestLazyImportContract:
    def test_import_omicverse_mol_is_free(self):
        """importing omicverse / omicverse.mol must not need the MD stack."""
        import omicverse as ov
        import omicverse.mol  # noqa: F401
        assert ov.mol is not None

    def test_public_md_surface_exists(self):
        import omicverse as ov
        for name in ("simulate", "prepare_system", "minimize", "equilibrate",
                     "production", "load_trajectory", "rmsd", "rmsf",
                     "radius_of_gyration", "hbonds", "contacts",
                     "native_contacts", "secondary_structure", "rmsd_matrix",
                     "cluster", "pca", "plot_rmsd", "plot_rmsf", "plot_rg",
                     "plot_energy", "mmgbsa", "alchemical_free_energy",
                     "MDSystem", "MDTrajectory", "MMGBSAResult"):
            assert hasattr(ov.mol, name), f"ov.mol.{name} missing"
            assert name in ov.mol.__all__, f"{name} not exported in __all__"

    def test_md_functions_are_registered(self):
        """The agent registry must be able to discover the MD capability."""
        from omicverse._registry import get_registry
        import omicverse.mol.md._run  # noqa: F401  (force registration)

        reg = get_registry()
        hits = reg.find("molecular dynamics")
        assert any("simulate" in str(h).lower() for h in hits), \
            "ov.mol.simulate is not discoverable in the registry"

    @pytest.mark.skipif(_HAS_OPENMM, reason="openmm IS installed here")
    def test_missing_backend_raises_actionable_error(self):
        import omicverse as ov
        with pytest.raises(ImportError) as exc:
            ov.mol.simulate("1UBQ", ns=0.001)
        msg = str(exc.value)
        assert "omicverse[md]" in msg and "conda" in msg

    def test_pdbfixer_is_vendored_not_a_dependency(self):
        """PDBFixer is not on PyPI, so it must ship inside omicverse.

        Importing the vendored package must also stay free of OpenMM — the
        data files have to be reachable even when the [md] extra is absent.
        """
        import omicverse.external.pdbfixer as vend      # must not need openmm
        base = os.path.dirname(vend.__file__)
        assert os.path.exists(os.path.join(base, "soft.xml"))
        assert os.path.exists(os.path.join(base, "templates", "ALA.pdb"))
        assert os.path.exists(os.path.join(base, "LICENSE"))
        assert vend.__vendored_version__ == "1.8.1"


# --------------------------------------------------------------------- #
# 2. CPU smoke test — seconds, no GPU needed
# --------------------------------------------------------------------- #
@needs_openmm
@needs_mdtraj
class TestCPUSmoke:
    def test_prepare_minimize_produce_implicit(self, dipeptide_pdb, tmp_path):
        import omicverse as ov

        sysobj = ov.mol.prepare_system(dipeptide_pdb, water="implicit",
                                       verbose=False)
        assert sysobj.n_atoms > 10
        assert not sysobj.is_periodic
        assert sysobj.provenance["implicit_solvent"] is True

        sysobj = ov.mol.minimize(sysobj, platform="CPU", verbose=False)
        stage = sysobj.provenance["stages"][-1]
        assert stage["stage"] == "minimize"
        e = stage["energy_after_kj_mol"]
        assert np.isfinite(e), "post-minimisation energy must be finite"
        assert e <= stage["energy_before_kj_mol"] + 1e-6

        traj_path = str(tmp_path / "prod.dcd")
        traj = ov.mol.production(sysobj, ns=0.002, report_ps=0.2,
                                 ensemble="NVT", traj=traj_path,
                                 platform="CPU", progress=False, verbose=False)
        assert os.path.exists(traj.traj_path)
        assert os.path.exists(traj.top_path)
        assert traj.n_frames == 10                # 2 ps / 0.2 ps

        loaded = traj.load()
        assert loaded.n_frames == traj.n_frames
        assert np.isfinite(loaded.xyz).all(), "trajectory contains NaN"

    def test_analysis_on_cpu_trajectory(self, dipeptide_pdb, tmp_path):
        import omicverse as ov

        s = ov.mol.prepare_system(dipeptide_pdb, water="implicit", verbose=False)
        s = ov.mol.minimize(s, platform="CPU", verbose=False)
        traj = ov.mol.production(s, ns=0.002, report_ps=0.2, ensemble="NVT",
                                 traj=str(tmp_path / "a.dcd"), platform="CPU",
                                 progress=False, verbose=False)

        r = ov.mol.rmsd(traj, selection="name CA")
        assert r.shape == (traj.n_frames,) and np.isfinite(r).all()
        rg = ov.mol.radius_of_gyration(traj)
        assert rg.shape == (traj.n_frames,) and (rg > 0).all()

    def test_timestep_guard_without_hmr(self, dipeptide_pdb, tmp_path):
        """4 fs without hydrogen-mass repartitioning must be refused."""
        import omicverse as ov
        s = ov.mol.prepare_system(dipeptide_pdb, water="implicit", verbose=False)
        with pytest.raises(ValueError, match="hydrogen-mass"):
            ov.mol.production(s, ns=0.001, timestep_fs=4.0,
                              traj=str(tmp_path / "x.dcd"), platform="CPU",
                              progress=False, verbose=False)

    def test_production_is_reproducible_on_cpu(self, dipeptide_pdb, tmp_path):
        """Same system + same seed + same platform reproduces a run exactly.

        This is the contract that actually holds: from one MDSystem, repeated
        production runs are identical on CPU. It also pins the decision that
        production does *not* mutate its input — if it wrote the final state
        back, the second call would silently continue the first.
        """
        import omicverse as ov

        s = ov.mol.prepare_system(dipeptide_pdb, water="implicit", seed=42,
                                  verbose=False)
        s = ov.mol.minimize(s, platform="CPU", verbose=False)

        def run(tag):
            t = ov.mol.production(s, ns=0.002, report_ps=0.5, ensemble="NVT",
                                  traj=str(tmp_path / f"{tag}.dcd"),
                                  platform="CPU", seed=42, progress=False,
                                  verbose=False)
            return t.load().xyz

        a, b = run("d1"), run("d2")
        assert np.allclose(a, b, atol=1e-5), \
            "same system + seed + platform must reproduce the run exactly"

    def test_prepare_system_is_seeded(self, dipeptide_pdb):
        """`seed` must control hydrogen placement / solvation randomness.

        OpenMM's Modeller draws from the global RNG, so without seeding two
        identical calls give different starting coordinates.
        """
        import random

        import omicverse as ov
        from openmm import unit

        def prep(seed):
            s = ov.mol.prepare_system(dipeptide_pdb, water="implicit",
                                      seed=seed, verbose=False)
            return np.array(s.positions.value_in_unit(unit.nanometer))

        a, b = prep(42), prep(42)
        assert np.abs(a - b).max() < 1e-3, "same seed must rebuild the same structure"
        assert not np.allclose(a, prep(7)), "a different seed should differ"

        # and it must not hijack the caller's RNG stream
        random.seed(123)
        first = random.random()
        prep(999)
        random.seed(123)
        assert random.random() == first

    def test_provenance_is_recorded(self, dipeptide_pdb, tmp_path):
        import omicverse as ov
        s = ov.mol.prepare_system(dipeptide_pdb, water="implicit", verbose=False)
        s = ov.mol.minimize(s, platform="CPU", verbose=False)
        traj = ov.mol.production(s, ns=0.001, report_ps=0.5, ensemble="NVT",
                                 traj=str(tmp_path / "p.dcd"), platform="CPU",
                                 seed=7, progress=False, verbose=False)
        p = traj.provenance
        assert p["engine"] == "openmm"
        assert p["versions"]["openmm"]
        assert p["integrator"] == "LangevinMiddle"
        assert p["seed"] == 7
        assert p["constraints"] == "HBonds"
        assert p["files"]["trajectory_sha256"]
        assert p["platform"]

    def test_mmgbsa_rejects_pbsa_and_ligandless(self, dipeptide_pdb, tmp_path):
        import omicverse as ov
        s = ov.mol.prepare_system(dipeptide_pdb, water="implicit", verbose=False)
        s = ov.mol.minimize(s, platform="CPU", verbose=False)
        traj = ov.mol.production(s, ns=0.001, report_ps=0.5, ensemble="NVT",
                                 traj=str(tmp_path / "m.dcd"), platform="CPU",
                                 progress=False, verbose=False)
        with pytest.raises(NotImplementedError, match="Poisson-Boltzmann"):
            ov.mol.mmgbsa(traj, method="pbsa")
        with pytest.raises(ValueError, match="no ligand"):
            ov.mol.mmgbsa(traj, frames=2, verbose=False)

    def test_alchemical_free_energy_is_documented_not_silent(self):
        import omicverse as ov
        with pytest.raises(NotImplementedError, match="openmmtools"):
            ov.mol.alchemical_free_energy()


# --------------------------------------------------------------------- #
# 3. GPU pipeline
# --------------------------------------------------------------------- #
@needs_openmm
@needs_mdtraj
@needs_gpu
class TestGPU:
    def test_platform_is_cuda_and_reports_speed(self, dipeptide_pdb, tmp_path):
        import omicverse as ov

        s = ov.mol.prepare_system(dipeptide_pdb, water="implicit", verbose=False)
        s = ov.mol.minimize(s, platform="CUDA", verbose=False)
        traj = ov.mol.production(s, ns=0.02, report_ps=1.0, ensemble="NVT",
                                 traj=str(tmp_path / "gpu.dcd"),
                                 platform="CUDA", progress=False, verbose=False)
        assert traj.provenance["platform"] == "CUDA"
        assert traj.provenance["precision"] == "mixed"
        # the state log must carry a ns/day column
        with open(traj.log_path) as fh:
            header = fh.readline()
        assert "Speed" in header or "ns/day" in header

    def test_explicit_solvent_full_pipeline(self, dipeptide_pdb, tmp_path):
        """prepare (explicit water + salt) -> minimize -> equilibrate -> production."""
        import omicverse as ov

        s = ov.mol.prepare_system(dipeptide_pdb, water="tip3p",
                                  box_padding_nm=1.2, ions="0.15 M NaCl",
                                  verbose=False)
        assert s.is_periodic and s.n_atoms > 500      # solvated
        s = ov.mol.minimize(s, platform="CUDA", verbose=False)
        s = ov.mol.equilibrate(s, nvt_ps=2, npt_ps=2, platform="CUDA",
                               seed=1, verbose=False)
        traj = ov.mol.production(s, ns=0.01, report_ps=1.0, traj=str(
            tmp_path / "npt.dcd"), platform="CUDA", seed=1, progress=False,
            verbose=False)
        assert traj.n_frames == 10
        loaded = traj.load()
        assert np.isfinite(loaded.xyz).all()

    def test_checkpoint_resume_continues(self, dipeptide_pdb, tmp_path):
        import omicverse as ov

        s = ov.mol.prepare_system(dipeptide_pdb, water="implicit", verbose=False)
        s = ov.mol.minimize(s, platform="CUDA", verbose=False)
        first = ov.mol.production(s, ns=0.004, report_ps=0.5, ensemble="NVT",
                                  traj=str(tmp_path / "r.dcd"),
                                  checkpoint_ps=0.5, platform="CUDA",
                                  progress=False, verbose=False)
        assert os.path.exists(first.checkpoint)
        second = ov.mol.production(s, ns=0.004, report_ps=0.5, ensemble="NVT",
                                   traj=str(tmp_path / "r2.dcd"),
                                   resume=first.checkpoint, platform="CUDA",
                                   progress=False, verbose=False)
        assert second.n_frames == first.n_frames
        assert second.provenance["resumed_from"] == first.checkpoint

    def test_gpu_and_cpu_agree_statistically(self, dipeptide_pdb, tmp_path):
        """A GPU run must be physically equivalent to a CPU one.

        Trajectories cannot match frame-by-frame across platforms (MD is
        chaotic, and GPU reductions are not order-deterministic), so compare
        an ensemble property instead: the radius of gyration distribution.
        """
        import omicverse as ov

        def run(tag, platform):
            s = ov.mol.prepare_system(dipeptide_pdb, water="implicit",
                                      verbose=False)
            s = ov.mol.minimize(s, platform=platform, verbose=False)
            return ov.mol.production(
                s, ns=0.01, report_ps=0.2, ensemble="NVT",
                traj=str(tmp_path / f"{tag}.dcd"), platform=platform,
                seed=7, progress=False, verbose=False)

        gpu = ov.mol.radius_of_gyration(run("gpu", "CUDA"))
        cpu = ov.mol.radius_of_gyration(run("cpu", "CPU"))
        assert np.isfinite(gpu).all() and np.isfinite(cpu).all()
        # same physics -> same mean size, well within thermal fluctuation
        assert abs(float(gpu.mean()) - float(cpu.mean())) < 0.05


@needs_openmm
@needs_mdtraj
@needs_gpu
@needs_network
class TestGPUProtein:
    def test_simulate_real_protein_and_analyse(self, tmp_path):
        """fetch_structure -> simulate -> analysis, the headline workflow."""
        import omicverse as ov

        s = ov.mol.fetch_structure("1UBQ", source="pdb")
        traj = ov.mol.simulate(s, ns=0.05, water="tip3p", box_padding_nm=0.9,
                               nvt_ps=5, npt_ps=5, report_ps=2.0,
                               outdir=str(tmp_path / "md"), platform="CUDA",
                               seed=0, verbose=False)
        assert traj.provenance["platform"] == "CUDA"
        assert traj.n_frames >= 20

        r = ov.mol.rmsd(traj)
        assert np.isfinite(r).all()
        assert r.max() < 0.5, f"CA-RMSD blew past 0.5 nm: {r.max():.3f}"

        f = ov.mol.rmsf(traj)
        assert f.shape[0] == 76                    # ubiquitin residues
        rg = ov.mol.radius_of_gyration(traj)
        assert 1.0 < float(rg.mean()) < 1.5        # ubiquitin Rg ~1.2 nm
        ss = ov.mol.secondary_structure(traj)
        assert ss.shape[0] == traj.n_frames
        hb = ov.mol.hbonds(traj, freq=0.5)
        assert len(hb) > 10
        ax = ov.mol.plot_rmsd(traj)
        assert ax is not None


@needs_openmm
@needs_mdtraj
@needs_ligand_ff
class TestLigandComplexAndMMGBSA:
    """Protein-ligand MD and MM-GBSA without needing the network or Vina."""

    def test_parameterise_ligand_and_score(self, dipeptide_pdb, tmp_path):
        import omicverse as ov

        apo = ov.mol.prepare_system(dipeptide_pdb, water="implicit",
                                    seed=42, verbose=False)
        cplx = ov.mol.prepare_system(dipeptide_pdb, water="implicit",
                                     ligand="c1ccccc1O", ligand_ff="gaff-2.11",
                                     seed=42, verbose=False)
        # the ligand really was added and parameterised
        assert cplx.n_atoms > apo.n_atoms
        assert cplx.provenance["ligand_ff"] == "gaff-2.11"
        assert "O" in cplx.provenance["ligand_smiles"]

        cplx = ov.mol.minimize(cplx, platform="CPU", verbose=False)
        traj = ov.mol.production(cplx, ns=0.004, report_ps=0.5,
                                 ensemble="NVT",
                                 traj=str(tmp_path / "cplx.dcd"),
                                 platform="CPU", seed=0, progress=False,
                                 verbose=False)

        res = ov.mol.mmgbsa(traj, frames=4, platform="CPU", verbose=False)
        assert np.isfinite(res.dG)
        assert res.n_frames == 4
        assert len(res.per_frame) == 4

        c = res.components
        # physics the decomposition must obey, whatever the magnitudes:
        assert c["vdw"] < 0, "van der Waals contact is always stabilising"
        assert c["nonpolar_solvation"] < 0, "burying surface is favourable"
        assert c["polar_solvation"] > 0, (
            "desolvating polar groups on binding costs energy")
        assert abs(res.dG - sum(c.values())) < 1e-6, "terms must sum to dG"
        assert "dG_kcal_mol" in res.to_frame().columns

    def test_mmgbsa_provenance(self, dipeptide_pdb, tmp_path):
        import omicverse as ov

        s = ov.mol.prepare_system(dipeptide_pdb, water="implicit",
                                  ligand="CCO", seed=42, verbose=False)
        s = ov.mol.minimize(s, platform="CPU", verbose=False)
        traj = ov.mol.production(s, ns=0.002, report_ps=0.5, ensemble="NVT",
                                 traj=str(tmp_path / "e.dcd"), platform="CPU",
                                 progress=False, verbose=False)
        res = ov.mol.mmgbsa(traj, frames=2, platform="CPU", verbose=False)
        p = res.provenance
        assert "GBn2" in p["method"]
        assert p["entropy_term"] is False        # documented limitation
        assert p["n_frames_scored"] == 2


@needs_openmm
@needs_mdtraj
@needs_gpu
@needs_ligand_ff
@needs_network
class TestDockToMDToMMGBSA:
    """The moat: static docking pose -> MD ensemble -> binding free energy."""

    def test_dock_simulate_mmgbsa(self, tmp_path):
        import omicverse as ov

        s = ov.mol.fetch_structure("1M17", source="pdb")
        ov.mol.pockets(s)          # detect pockets before docking into one
        # erlotinib given by SMILES rather than by name, so the test docks the
        # real EGFR inhibitor without depending on the ChEMBL service being up.
        erlotinib = "CCOCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
        result = ov.mol.dock(s, erlotinib, pocket=1, seed=42)
        assert result.receptor is not None, \
            "DockingResult must carry the receptor for the MD handoff"

        traj = ov.mol.simulate(result, ns=0.05, water="implicit",
                               nvt_ps=5, npt_ps=0, report_ps=2.0,
                               outdir=str(tmp_path / "cplx"), platform="CUDA",
                               seed=0, verbose=False)
        assert traj.n_frames >= 10

        res = ov.mol.mmgbsa(traj, frames=5, platform="CUDA", verbose=False)
        assert np.isfinite(res.dG)
        assert -150.0 < res.dG < 20.0, f"implausible dG: {res.dG}"
        for term in ("vdw", "electrostatic", "polar_solvation",
                     "nonpolar_solvation"):
            assert term in res.components and np.isfinite(res.components[term])
        assert res.components["vdw"] < 0            # vdW always stabilising
        df = res.to_frame()
        assert "dG_kcal_mol" in df.columns

        n_contacts = ov.mol.contacts(traj, "protein", "resname LIG")
        assert n_contacts.shape[0] == traj.n_frames
