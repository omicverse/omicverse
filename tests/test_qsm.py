"""Tests for omicverse.tl._qsm (quantitative susceptibility mapping), on a numerical phantom."""
import numpy as np
import pytest

from omicverse.tl._qsm import (
    GAMMA_BAR_MHZ_PER_T, qsm_background_removal, qsm_brain_mask, qsm_dipole_inversion,
    qsm_dipole_kernel, qsm_forward_field, qsm_load_multi_echo, qsm_multi_echo_field,
    qsm_numerical_phantom, qsm_reconstruct, qsm_save_nifti, qsm_unwrap_phase,
)


@pytest.fixture(scope="module")
def phantom():
    return qsm_numerical_phantom(shape=(72, 72, 56), b0_tesla=3.0,
                                 echo_times=(0.004, 0.008, 0.012, 0.016), noise=0.01, seed=1)


def _nuclei_slope(P, chi, region):
    tv, pv = [], []
    for name, (s, v) in P["structures"].items():
        if not name.startswith("nucleus"):
            continue
        s2 = s & region
        if s2.sum() > 5:
            tv.append(v)
            pv.append(float(chi[s2].mean()))
    return np.polyfit(tv, pv, 1)[0], np.corrcoef(tv, pv)[0, 1]


def test_dipole_kernel_values():
    d = qsm_dipole_kernel((16, 16, 16), (1, 1, 1), (0, 0, 1))
    assert d[0, 0, 0] == 0
    assert np.isclose(d[0, 0, 1], -2 / 3, atol=1e-6)      # k parallel to B0
    assert np.isclose(d[1, 0, 0], 1 / 3, atol=1e-6)       # k perpendicular
    # oblique B0 direction is normalised and rotates the kernel
    d2 = qsm_dipole_kernel((16, 16, 16), (1, 1, 1), (0, 0, 5))
    assert np.allclose(d, d2)
    d3 = qsm_dipole_kernel((16, 16, 16), (1, 1, 1), (1, 0, 0))
    assert np.isclose(d3[1, 0, 0], -2 / 3, atol=1e-6)


def test_forward_field_sphere_pattern():
    n = 48
    g = np.meshgrid(*[np.arange(n) - n / 2] * 3, indexing="ij")
    chi = ((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) <= 16).astype(np.float32)
    f = qsm_forward_field(chi, (1, 1, 1), (0, 0, 1), pad_mm=16)
    c = n // 2
    # outside a positive sphere: positive along B0, negative (half as large) across
    along, across = f[c, c, c + 8], f[c + 8, c, c]
    assert along > 0 > across
    assert np.isclose(along / -across, 2.0, rtol=0.15)
    # inside a uniform sphere the Lorentzian-corrected field is ~0
    assert abs(f[c, c, c]) < 0.05


def test_unwrap_recovers_smooth_phase():
    n = 40
    g = np.meshgrid(*[np.linspace(-1, 1, n)] * 3, indexing="ij")
    true = 12 * (g[0] ** 2 + 0.5 * g[1] - 0.3 * g[2] ** 2)
    wrapped = np.angle(np.exp(1j * true))
    mask = (g[0] ** 2 + g[1] ** 2 + g[2] ** 2) < 0.8
    u = qsm_unwrap_phase(wrapped, mask, method="quality")
    diff = (u - true)[mask]
    diff -= 2 * np.pi * np.round(np.median(diff) / (2 * np.pi))
    assert np.mean(np.abs(diff) < 1e-3) > 0.99
    # Laplacian unwrapping is approximate on steep phase but must stay 2*pi-consistent
    # with the measurement and follow the true phase on a gentle one
    gentle = 3 * (g[0] ** 2 + 0.5 * g[1] - 0.3 * g[2] ** 2)
    ul = qsm_unwrap_phase(np.angle(np.exp(1j * gentle)), mask, method="laplacian")
    k = (ul - np.angle(np.exp(1j * gentle)))[mask] / (2 * np.pi)
    assert np.allclose(k, np.round(k), atol=1e-4)
    assert np.corrcoef(ul[mask], gentle[mask])[0, 1] > 0.99


def test_multi_echo_field_recovers_total_field(phantom):
    P = phantom
    m = P["brain_mask"]
    fe = qsm_multi_echo_field(P["phase"], P["magnitude"], P["echo_times"], m)
    err = (fe["field_hz"] - P["field_hz"])[m]
    err -= np.median(err)                       # a global 2*pi/TE ambiguity is harmless
    assert np.percentile(np.abs(err), 95) < 1.0  # Hz
    assert fe["reliable_mask"].sum() > 0.9 * m.sum()
    # milliseconds are recognised
    fe_ms = qsm_multi_echo_field(P["phase"], P["magnitude"], np.asarray(P["echo_times"]) * 1000, m)
    assert np.allclose(fe_ms["field_hz"], fe["field_hz"], atol=1e-3)


def test_brain_mask_covers_brain(phantom):
    m = qsm_brain_mask(phantom["magnitude"], phantom["voxel_size"])
    b = phantom["brain_mask"]
    assert (m & b).sum() > 0.98 * b.sum()
    assert not m[0, 0, 0]


@pytest.mark.parametrize("method", ["vsharp", "pdf"])
def test_background_removal_matches_internal_sources(phantom, method):
    P = phantom
    m = P["brain_mask"]
    loc, lm = qsm_background_removal(P["field_hz"], m, P["voxel_size"], method=method,
                                     b0_dir=P["b0_dir"])
    truth = qsm_forward_field(np.where(m, P["chi"], 0), P["voxel_size"], P["b0_dir"], pad_mm=16)
    truth = truth * GAMMA_BAR_MHZ_PER_T * P["b0_tesla"]
    a, t = loc[lm] - loc[lm].mean(), truth[lm] - truth[lm].mean()
    assert np.corrcoef(a, t)[0, 1] > 0.9
    # the external (air) field is orders of magnitude larger than what is left
    assert np.std(a - t) < 0.2 * np.std(P["field_hz"][m])


@pytest.mark.parametrize("method", ["tv", "tikhonov", "tkd"])
def test_inversion_on_exact_local_field(phantom, method):
    P = phantom
    m = P["brain_mask"]
    local = qsm_forward_field(np.where(m, P["chi"], 0), P["voxel_size"], P["b0_dir"], pad_mm=16)
    chi = qsm_dipole_inversion(local, m, P["voxel_size"], P["b0_dir"], method=method)
    slope, r = _nuclei_slope(P, chi, m)
    assert r > 0.95
    assert 0.6 < slope < 1.2, slope
    assert np.all(chi[~m] == 0)


def test_reconstruct_end_to_end_and_units(phantom, tmp_path):
    P = phantom
    res = qsm_reconstruct(P["phase"], P["magnitude"], P["echo_times"], P["b0_tesla"],
                          P["voxel_size"], b0_dir=P["b0_dir"])
    chi, lm = res["chi"], res["local_mask"]
    slope, r = _nuclei_slope(P, chi, lm)
    assert r > 0.95 and 0.6 < slope < 1.2, (slope, r)
    # wrong field strength scales the map: ppm conversion uses gamma_bar * B0
    res2 = qsm_reconstruct(P["phase"], P["magnitude"], P["echo_times"], 2 * P["b0_tesla"],
                           P["voxel_size"], b0_dir=P["b0_dir"], inversion="tikhonov")
    res1 = qsm_reconstruct(P["phase"], P["magnitude"], P["echo_times"], P["b0_tesla"],
                           P["voxel_size"], b0_dir=P["b0_dir"], inversion="tikhonov")
    assert np.allclose(res2["chi"], res1["chi"] / 2, atol=1e-5)
    assert set(res["summary"]) >= {"dipole_inconsistency", "mask_voxels", "chi_ppm_p1_p50_p99"}
    # NIfTI round trip with geometry
    aff = np.diag([0.8, 0.9, 1.1, 1.0])
    path = tmp_path / "chi.nii"
    qsm_save_nifti(chi, str(path), affine=aff)
    from omicverse.tl._qsm import _read_nifti, _read_nifti_minimal
    for reader in (_read_nifti, _read_nifti_minimal):
        back, aff2, zooms = reader(str(path))
        assert back.shape == chi.shape
        assert np.allclose(back, chi, atol=1e-6)
        assert np.allclose(zooms, (0.8, 0.9, 1.1), atol=1e-5)


def test_load_multi_echo_bids(tmp_path, phantom):
    import json
    P = phantom
    for e, t in enumerate(P["echo_times"], start=1):
        for part, arr in (("phase", P["phase"][e - 1]), ("mag", P["magnitude"][e - 1])):
            stem = tmp_path / f"sub-x_echo-{e}_part-{part}_MEGRE"
            qsm_save_nifti(arr, str(stem) + ".nii", affine=np.eye(4))
            with open(str(stem) + ".json", "w") as fh:
                json.dump({"EchoTime": float(t), "MagneticFieldStrength": 3}, fh)
    d = qsm_load_multi_echo(str(tmp_path))
    assert d["phase"].shape == P["phase"].shape
    assert np.allclose(d["echo_times"], P["echo_times"])
    assert d["b0_tesla"] == 3.0
    assert np.allclose(d["b0_dir"], (0, 0, 1))
