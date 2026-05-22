"""Integration tests for the ``ov.mol`` module.

Covers:
- Lazy import + module surface (fetch_structure, predict_structure, view,
  pockets, druggability, known_drugs, dock, redock_validate, ...)
- ``@register_function`` metadata visible in the omicverse registry
- ``_check`` dependency gates raise actionable ImportErrors
- Pure-Python helpers (search-box geometry, identifier regexes)
- Network-backed smoke tests (RCSB PDB + AlphaFold DB fetch, py3Dmol view,
  PAE plot) — these skip gracefully when offline.

The structural-biology stack (biotite, py3Dmol) is the ``omicverse[mol]``
optional dependency — the whole file skips if it is not installed.
"""
from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

_CORE = ["biotite", "py3Dmol"]
_MISSING = [m for m in _CORE if importlib.util.find_spec(m) is None]
pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason=f"ov.mol core packages not installed: {_MISSING}",
)

_HAS_DOCK = all(importlib.util.find_spec(m) is not None
                for m in ("rdkit", "vina", "meeko"))
_HAS_FPOCKET = importlib.util.find_spec("fpocket_rs") is not None


def _net(fn, *args, **kwargs):
    """Run a network call; skip the test if the network is unavailable."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on connectivity
        import requests
        if isinstance(exc, (requests.exceptions.ConnectionError,
                            requests.exceptions.Timeout)):
            pytest.skip(f"network unavailable: {exc}")
        raise


# --------------------------------------------------------------------- #
# module surface + registry                                             #
# --------------------------------------------------------------------- #


def test_module_loads_and_exposes_public_api():
    import omicverse as ov

    surface = ["MolStructure", "fetch_structure", "predict_structure",
               "view", "view_docking", "plot_pae", "pockets",
               "druggability", "known_drugs", "dock", "redock_validate",
               "DockingResult"]
    for name in surface:
        assert hasattr(ov.mol, name), f"ov.mol missing {name}"


def test_registry_sees_mol_module():
    import omicverse as ov  # noqa: F401
    from omicverse._registry import get_registry

    ov.mol._hydrate_registry()
    reg = get_registry()
    names = {e["short_name"] for e in reg.get_by_category("mol")}
    for fn in ("fetch_structure", "view", "pockets", "known_drugs", "dock"):
        assert fn in names, f"{fn} not registered under category 'mol'"


# --------------------------------------------------------------------- #
# dependency gates                                                       #
# --------------------------------------------------------------------- #


def test_check_need_raises_actionable_error():
    from omicverse.mol._check import _need

    with pytest.raises(ImportError) as exc:
        _need("a_package_that_does_not_exist", "mol", "a feature")
    msg = str(exc.value)
    assert "omicverse[mol]" in msg and "a feature" in msg


# --------------------------------------------------------------------- #
# pure-Python helpers (no network)                                       #
# --------------------------------------------------------------------- #


def test_box_from_coords_is_centroid_centred_and_encloses():
    from omicverse.mol._dock import _box_from_coords

    # asymmetric point cloud: a dense core plus a far outlier
    coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                       [1., 1., 0.], [20., 0., 0.]])
    center, size = _box_from_coords(coords, padding=4.0, minimum=10.0)
    # centred on the centroid, not the bounding-box midpoint
    assert center[0] == pytest.approx(coords[:, 0].mean())
    # large enough to enclose the full extent plus padding
    assert size[0] >= (coords[:, 0].ptp() + 8.0) - 1e-6
    # the per-axis minimum is honoured
    assert min(size) >= 10.0 - 1e-6


def test_identifier_regexes():
    from omicverse.mol._structure import _PDB_RE, _UNIPROT_RE

    assert _PDB_RE.match("1M17") and _PDB_RE.match("1crn")
    assert not _PDB_RE.match("EGFR")
    assert _UNIPROT_RE.match("P00533") and _UNIPROT_RE.match("Q9Y2X7")
    assert not _UNIPROT_RE.match("EGFR")


# --------------------------------------------------------------------- #
# network-backed smoke tests                                             #
# --------------------------------------------------------------------- #


def test_fetch_experimental_structure_and_view(tmp_path):
    import omicverse as ov

    # crambin — a tiny 46-residue experimental structure
    s = _net(ov.mol.fetch_structure, "1CRN", source="pdb",
             dir=str(tmp_path))
    assert s.source == "pdb"
    assert s.n_residues == 46
    assert len(s.sequence) == 46
    assert s.sequence.startswith("TTCCPSIV")

    v = ov.mol.view(s, color_by="chain", width=320, height=240)
    html = v._make_html()
    assert "3dmol" in html.lower() and len(html) > 1000


def test_fetch_alphafold_model_has_plddt_and_pae(tmp_path):
    import omicverse as ov

    # insulin (P01308) — a small AlphaFold DB model
    s = _net(ov.mol.fetch_structure, "P01308", dir=str(tmp_path))
    assert s.source == "alphafold"
    assert s.uniprot == "P01308"
    assert s.plddt is not None and s.plddt.shape[0] == s.n_residues
    # pLDDT is on the 0-100 scale
    assert 0.0 <= float(np.min(s.plddt)) and float(np.max(s.plddt)) <= 100.0
    assert s.pae is not None
    assert s.pae.shape == (s.n_residues, s.n_residues)


def test_plot_pae_returns_axes(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import omicverse as ov

    s = _net(ov.mol.fetch_structure, "P01308", dir=str(tmp_path))
    ax = ov.mol.plot_pae(s)
    assert ax.images, "plot_pae drew no heatmap"


def test_view_color_by_per_residue_score(tmp_path):
    import omicverse as ov

    s = _net(ov.mol.fetch_structure, "1CRN", source="pdb",
             dir=str(tmp_path))
    score = {int(r): float(i) for i, r in enumerate(s.residue_ids)}
    v = ov.mol.view(s, color_by=score, width=320, height=240)
    assert "3dmol" in v._make_html().lower()


def test_view_color_by_array_wrong_length_raises(tmp_path):
    import omicverse as ov

    s = _net(ov.mol.fetch_structure, "1CRN", source="pdb",
             dir=str(tmp_path))
    with pytest.raises(ValueError):
        ov.mol.view(s, color_by=np.zeros(5))


# --------------------------------------------------------------------- #
# pocket / docking layers (optional backends)                            #
# --------------------------------------------------------------------- #


@pytest.mark.skipif(not _HAS_FPOCKET, reason="fpocket-rs not installed")
def test_pockets_and_druggability(tmp_path):
    import omicverse as ov

    s = _net(ov.mol.fetch_structure, "1CRN", source="pdb",
             dir=str(tmp_path))
    df = ov.mol.pockets(s)
    assert list(df.columns)[:4] == ["pocket_id", "rank", "drug_score",
                                    "volume"]
    assert s.pockets is df
    verdict = ov.mol.druggability(s)
    assert set(verdict) >= {"top_drug_score", "druggable", "verdict",
                            "n_pockets", "pockets"}


@pytest.mark.skipif(not _HAS_DOCK, reason="omicverse[mol-dock] not installed")
def test_dock_pipeline_runs(tmp_path):
    import omicverse as ov

    s = _net(ov.mol.fetch_structure, "1CRN", source="pdb",
             dir=str(tmp_path))
    # blind dock a tiny rigid ligand, minimal exhaustiveness
    result = ov.mol.dock(s, "c1ccccc1", exhaustiveness=1, n_poses=3,
                         seed=1)
    assert len(result.poses) >= 1
    assert len(result.affinities) == len(result.poses)
    assert len(result.pose_blocks) == len(result.poses)
    assert result.best is not None
