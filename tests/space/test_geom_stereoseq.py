"""Geometry and Stereo-seq I/O.

The alignment tests work by imposing a known rigid transform on a real slide and
asking the aligner to undo it. That is the only way to know an alignment is right
rather than merely smooth-looking: a wrong rotation still produces a tidy picture.

Measured when this file was written, on a Visium slide 10,628 px across:

    fgw   exact (< 1e-11 px) at every rotation from 10 to 170 degrees
    icp   exact below ~60 degrees, then fails outright (3.6e+03 px at 79 degrees)

which is why fgw is the default.
"""
from __future__ import annotations

import builtins
import gzip
import sys

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from omicverse import space
from omicverse.io import spatial as io_spatial


def _slide(n: int = 250, seed: int = 0) -> AnnData:
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1000, size=(n, 2))
    adata = AnnData(rng.poisson(4, size=(n, 25)).astype(np.float32))
    adata.obsm["spatial"] = coords
    adata.var_names = [f"g{i}" for i in range(25)]
    return adata


def _rotate(adata: AnnData, degrees: float, shift=(0.0, 0.0)) -> AnnData:
    t = np.deg2rad(degrees)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    xy = np.asarray(adata.obsm["spatial"], dtype=float)
    out = adata.copy()
    out.obsm["spatial"] = (xy - xy.mean(0)) @ rot + xy.mean(0) + np.asarray(shift)
    return out


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_procrustes_recovers_a_rotation_and_never_reflects():
    from omicverse.space._geom import _procrustes

    rng = np.random.default_rng(0)
    src = rng.normal(size=(150, 2)) * 100
    for degrees in (25, -60, 137):
        t = np.deg2rad(degrees)
        truth = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        rot = _procrustes(src, src @ truth, np.ones(len(src)))
        moved = (src - src.mean(0)) @ rot
        target = src @ truth
        assert np.allclose(moved, target - target.mean(0), atol=1e-8)
        assert np.linalg.det(rot) > 0

    # a mirrored target must not be matched by mirroring the section
    mirrored = src @ np.array([[1.0, 0.0], [0.0, -1.0]])
    assert np.linalg.det(_procrustes(src, mirrored, np.ones(len(src)))) > 0


@pytest.mark.parametrize("degrees", [5, 10])
def test_icp_undoes_a_small_rotation(degrees):
    """ICP only works near where it starts, and how near depends on the tissue.

    On a real Visium slide with an irregular outline it held to 60 degrees; on
    this uniform random cloud it fails from about 20; on a regular lattice it
    fails at 10, because six-fold symmetry makes nearest-point correspondences
    genuinely ambiguous. The fixture here is the random cloud, so the test only
    claims the range that fixture supports.
    """
    ref = _slide()
    moved, _ = space.geom.align_pairwise(_rotate(ref, degrees), ref, method="icp")
    assert np.abs(moved - ref.obsm["spatial"]).max() < 1e-6


def test_icp_gives_up_on_a_large_rotation_where_fgw_does_not():
    """The documented failure, pinned so the docstring cannot drift from reality."""
    ref = _slide()
    turned = _rotate(ref, 90)
    icp, _ = space.geom.align_pairwise(turned, ref, method="icp")
    fgw, _ = space.geom.align_pairwise(turned, ref, method="fgw")
    span = np.ptp(ref.obsm["spatial"])
    assert np.abs(icp - ref.obsm["spatial"]).max() > 0.1 * span
    assert np.abs(fgw - ref.obsm["spatial"]).max() < 1e-4


@pytest.mark.parametrize("degrees", [15, 79, 150])
def test_fgw_undoes_a_rotation_at_any_angle(degrees):
    ref = _slide()
    moved, plan = space.geom.align_pairwise(_rotate(ref, degrees), ref, method="fgw")
    assert np.abs(moved - ref.obsm["spatial"]).max() < 1e-4
    assert plan.shape == (ref.n_obs, ref.n_obs)


def test_align_carries_the_rotation_along_the_chain():
    """Every section must land in the reference frame, not just the first."""
    ref = _slide()
    slices = [ref, _rotate(ref, 17, (60, -40)), _rotate(ref, -31, (-100, 75)),
              _rotate(ref, 48, (30, 150))]
    aligned = space.geom.align(slices, method="fgw")
    frame = aligned[0].obsm["spatial_aligned"]
    for section in aligned:
        assert np.abs(section.obsm["spatial_aligned"] - frame).max() < 1e-3


def test_align_from_a_middle_reference_walks_both_ways():
    ref = _slide()
    slices = [_rotate(ref, -20), ref, _rotate(ref, 25)]
    aligned = space.geom.align(slices, method="fgw", reference=1)
    frame = aligned[1].obsm["spatial_aligned"]
    for section in aligned:
        assert np.abs(section.obsm["spatial_aligned"] - frame).max() < 1e-3


def test_align_rejects_a_single_section_and_a_bad_reference():
    ref = _slide()
    with pytest.raises(ValueError, match="at least two"):
        space.geom.align([ref])
    with pytest.raises(IndexError, match="reference"):
        space.geom.align([ref, ref], reference=5)


def test_align_pairwise_rejects_an_unknown_method():
    ref = _slide()
    with pytest.raises(ValueError, match="method must be"):
        space.geom.align_pairwise(ref, ref, method="magic")


def test_fgw_missing_dependency_names_pot(monkeypatch):
    ref = _slide(n=12)
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] == "ot":
            raise ImportError("blocked import: ot")
        return original_import(name, globals, locals, fromlist, level)

    for name in [key for key in sys.modules if key == "ot" or key.startswith("ot.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match='pip install POT'):
        space.geom.align_pairwise(ref, ref, method="fgw")


def test_stack_gives_every_section_its_own_z():
    ref = _slide()
    aligned = space.geom.align([ref, _rotate(ref, 12), _rotate(ref, -9)], method="fgw")
    volume = space.geom.stack(aligned, z_spacing=10.0)
    assert volume.n_obs == 3 * ref.n_obs
    assert volume.obsm["spatial_3d"].shape[1] == 3
    assert sorted(set(volume.obsm["spatial_3d"][:, 2])) == [0.0, 10.0, 20.0]
    assert volume.uns["geom_stack"]["n_sections"] == 3


def test_stack_accepts_uneven_section_depths():
    ref = _slide()
    aligned = space.geom.align([ref, _rotate(ref, 12), _rotate(ref, -9)], method="fgw")
    volume = space.geom.stack(aligned, z_values=[0.0, 8.0, 25.0])
    assert sorted(set(volume.obsm["spatial_3d"][:, 2])) == [0.0, 8.0, 25.0]


def test_stack_says_which_step_was_skipped():
    ref = _slide()
    with pytest.raises(KeyError, match="geom.align"):
        space.geom.stack([ref, ref], z_spacing=1.0)


def test_interpolate_sits_between_its_two_neighbours():
    ref = _slide()
    aligned = space.geom.align([ref, _rotate(ref, 10, (50, 50))], method="fgw")
    mid = space.geom.interpolate(aligned[0], aligned[1], fraction=0.5)
    assert mid.n_obs == ref.n_obs
    assert mid.uns["geom_interpolated"] is True
    lo = aligned[0].obsm["spatial_aligned"]
    hi = aligned[1].obsm["spatial_aligned"]
    got = mid.obsm["spatial_aligned"]
    assert got.min() >= min(lo.min(), hi.min()) - 1e-6
    assert got.max() <= max(lo.max(), hi.max()) + 1e-6


def test_interpolate_rejects_a_fraction_outside_the_gap():
    ref = _slide()
    aligned = space.geom.align([ref, _rotate(ref, 10)], method="fgw")
    with pytest.raises(ValueError, match="fraction"):
        space.geom.interpolate(aligned[0], aligned[1], fraction=1.5)


# --------------------------------------------------------------------------- #
# Stereo-seq
# --------------------------------------------------------------------------- #
def _gem(tmp_path, n=4000, extent=400, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "geneID": rng.choice([f"Gene{i}" for i in range(30)], n),
        "x": rng.integers(0, extent, n),
        "y": rng.integers(0, extent, n),
        "MIDCount": rng.integers(1, 4, n),
    })
    path = tmp_path / "sample.gem.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("#FileFormat=GEMv0.1\n#StereoChip=SS200000\n")
        frame.to_csv(handle, sep="\t", index=False)
    return path, frame


def test_read_stereoseq_conserves_every_count(tmp_path):
    path, frame = _gem(tmp_path)
    adata = io_spatial.read_stereoseq(path, bin_size=50)
    assert int(adata.X.sum()) == int(frame["MIDCount"].sum())


def test_read_stereoseq_bins_the_field_as_asked(tmp_path):
    path, _ = _gem(tmp_path, extent=400)
    adata = io_spatial.read_stereoseq(path, bin_size=50)
    assert adata.n_obs == (400 // 50) ** 2
    assert adata.uns["stereoseq"]["bin_size"] == 50
    assert adata.uns["stereoseq"]["bin_size_um"] == 25.0


def test_read_stereoseq_keeps_the_gem_header(tmp_path):
    path, _ = _gem(tmp_path)
    adata = io_spatial.read_stereoseq(path, bin_size=50)
    assert adata.uns["stereoseq"]["header"]["StereoChip"] == "SS200000"


def test_rebinning_equals_reading_at_the_coarser_size(tmp_path):
    """The cheap path and the expensive path must agree, or one of them is wrong."""
    path, _ = _gem(tmp_path, extent=400)
    fine = io_spatial.read_stereoseq(path, bin_size=50)
    direct = io_spatial.read_stereoseq(path, bin_size=100)
    rebinned = io_spatial.bin_stereoseq(fine, bin_size=100)
    assert rebinned.shape == direct.shape
    assert int(rebinned.X.sum()) == int(direct.X.sum())


def test_rebinning_refuses_a_size_the_bins_do_not_nest_into(tmp_path):
    path, _ = _gem(tmp_path)
    fine = io_spatial.read_stereoseq(path, bin_size=50)
    with pytest.raises(ValueError, match="whole multiple"):
        io_spatial.bin_stereoseq(fine, bin_size=75)
    with pytest.raises(ValueError, match="coarser"):
        io_spatial.bin_stereoseq(fine, bin_size=25)


def test_read_stereoseq_filters_on_counts_and_genes(tmp_path):
    path, _ = _gem(tmp_path)
    loose = io_spatial.read_stereoseq(path, bin_size=20)
    strict = io_spatial.read_stereoseq(path, bin_size=20, min_counts=15, min_genes=5)
    assert strict.n_obs < loose.n_obs
    assert np.asarray(strict.X.sum(axis=1)).ravel().min() >= 15


def test_read_stereoseq_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        io_spatial.read_stereoseq(tmp_path / "absent.gem")
