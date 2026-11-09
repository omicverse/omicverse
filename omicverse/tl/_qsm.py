r"""Quantitative susceptibility mapping (QSM) from multi-echo gradient-echo MRI.

Pure numpy / scipy (scikit-image for quality-guided phase unwrapping when it
is installed).  Every step takes plain arrays, so it works for any scanner,
field strength, resolution and orientation; acquisition parameters (echo
times, field strength, B0 direction, voxel size) are arguments, never
constants of a dataset.

Pipeline (each step is a separate function so it can be checked on its own)

``qsm_load_multi_echo``
    BIDS-style ``*_echo-<n>_part-{mag,phase}*.nii[.gz]`` + JSON sidecars ->
    arrays, echo times (s), field strength, B0 direction, voxel size, affine.
``qsm_brain_mask``
    magnitude threshold + morphological opening -> largest connected
    component -> hole filling (a generous mask; refine it with phase quality).
``qsm_unwrap_phase``
    3D spatial unwrapping: quality-guided path following (Herraez 2002,
    scikit-image) or Laplacian (Schofield & Zhu 2003).
``qsm_multi_echo_field``
    phase-offset removal (MCPC-3D-S, Eckstein 2018), per-echo spatial
    unwrapping, 2*pi alignment between echoes, magnitude^2-weighted
    least-squares frequency fit through the origin -> total field in Hz, fit
    residual and a phase-reliability mask.
``qsm_background_removal``
    V-SHARP (Wu 2012; QSM.jl default radii), SHARP (Schweser 2011) or PDF
    (Liu 2011) -> local field and the mask it is valid in.
``qsm_dipole_inversion``
    TV-regularised ADMM (Bilgic 2014), Tikhonov (identity / gradient,
    closed form) or TKD (Shmueli 2009).
``qsm_reconstruct``
    the whole chain, Hz -> ppm conversion included, with a QC summary.
``qsm_numerical_phantom`` / ``qsm_forward_field`` / ``qsm_dipole_kernel``
    a numerical head phantom with known susceptibility and the forward model,
    for validating any pipeline before it is applied to real data.
``qsm_save_nifti``
    writes a map with the geometry (affine / voxel size) of the input.

Units: phase in radians, echo times in seconds, field in Hz (total field) or
ppm (local field), susceptibility in ppm.  ppm = Hz / (gamma_bar * B0[T]),
gamma_bar = 42.57747892 MHz/T (proton).
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import re
import warnings
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi

from .._registry import register_function

__all__ = [
    "qsm_load_multi_echo", "qsm_save_nifti", "qsm_brain_mask", "qsm_unwrap_phase",
    "qsm_multi_echo_field", "qsm_background_removal", "qsm_dipole_inversion",
    "qsm_reconstruct", "qsm_dipole_kernel", "qsm_forward_field",
    "qsm_numerical_phantom",
]

#: proton gyromagnetic ratio / 2*pi in MHz/T (CODATA 2018)
GAMMA_BAR_MHZ_PER_T = 42.577478518

_WORKERS = -1


# ── small helpers ────────────────────────────────────────────────────────────

def _vox3(voxel_size) -> Tuple[float, float, float]:
    v = np.broadcast_to(np.asarray(voxel_size, dtype=float), (3,))
    if np.any(v <= 0):
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    return tuple(float(x) for x in v)


def _unit(b0_dir) -> np.ndarray:
    b = np.asarray(b0_dir, dtype=float).reshape(3)
    n = np.linalg.norm(b)
    if n == 0:
        raise ValueError("b0_dir must be a non-zero 3-vector")
    return b / n


def _fftn(x):
    return sfft.fftn(x, workers=_WORKERS)


def _ifftn_real(x):
    return sfft.ifftn(x, workers=_WORKERS).real


def _ball(radius_mm: float, voxel_size) -> np.ndarray:
    vs = _vox3(voxel_size)
    r = [max(int(np.floor(radius_mm / v)), 0) for v in vs]
    g = np.meshgrid(*[np.arange(-ri, ri + 1) * v for ri, v in zip(r, vs)], indexing="ij")
    return (g[0] ** 2 + g[1] ** 2 + g[2] ** 2) <= radius_mm ** 2 + 1e-9


def _largest_component(m: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(m)
    if n <= 1:
        return m.astype(bool)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def _otsu(x: np.ndarray, bins: int = 256) -> float:
    x = x[np.isfinite(x)]
    hist, edges = np.histogram(x, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist).astype(float)
    w1 = w0[-1] - w0
    c = np.cumsum(hist * mids)
    m0 = c / np.maximum(w0, 1)
    m1 = (c[-1] - c) / np.maximum(w1, 1)
    return float(mids[int(np.argmax((w0 * w1 * (m0 - m1) ** 2)[:-1]))])


def _echo_times_seconds(echo_times) -> np.ndarray:
    te = np.asarray(echo_times, dtype=float).ravel()
    if np.any(te <= 0):
        raise ValueError("echo times must be positive")
    if te.min() > 0.5:          # clearly milliseconds (no GRE echo is 0.5 s)
        warnings.warn("echo_times look like milliseconds; converting to seconds")
        te = te / 1000.0
    return te


def _kgrid(shape, voxel_size):
    vs = _vox3(voxel_size)
    ks = [sfft.fftfreq(n, d=v) for n, v in zip(shape, vs)]
    return np.meshgrid(*ks, indexing="ij", sparse=True)


def _pad(x, pad):
    if not pad:
        return x
    return np.pad(x, [(p, p) for p in pad], mode="constant")


def _unpad(x, pad):
    if not pad:
        return x
    return x[tuple(slice(p, x.shape[i] - p) for i, p in enumerate(pad))]


def _pad_width(shape, voxel_size, pad_mm):
    if not pad_mm:
        return None
    vs = _vox3(voxel_size)
    return [int(np.ceil(pad_mm / v)) for v in vs]


# ── kernels and forward model ───────────────────────────────────────────────

@register_function(
    aliases=["偶极子核", "dipole kernel", "QSM dipole kernel", "unit dipole k-space"],
    category="utils",
    description="k-space unit dipole kernel D(k) = 1/3 - (k.b)^2/|k|^2 (D(0)=0) for a grid shape, "
                "voxel size (mm) and B0 direction in voxel coordinates.",
    requires={}, produces={}, auto_fix="none",
    examples=["D = ov.tl.qsm_dipole_kernel(chi.shape, (1, 1, 1), b0_dir=(0, 0, 1))"],
)
def qsm_dipole_kernel(shape, voxel_size=(1.0, 1.0, 1.0), b0_dir=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Unit dipole kernel on the unshifted FFT grid (float32)."""
    kx, ky, kz = _kgrid(shape, voxel_size)
    b = _unit(b0_dir)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    kb = kx * b[0] + ky * b[1] + kz * b[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1.0 / 3.0 - kb ** 2 / k2
    d[k2 == 0] = 0.0
    return d.astype(np.float32)


@register_function(
    aliases=["QSM 正演", "磁化率正演场", "forward field from susceptibility", "dipole convolution"],
    category="utils",
    description="Forward model of QSM: field perturbation (ppm of B0) produced by a susceptibility "
                "map (ppm), by dipole convolution in k-space with zero padding.",
    requires={}, produces={}, auto_fix="none",
    examples=["field_ppm = ov.tl.qsm_forward_field(chi, voxel_size, b0_dir)"],
)
def qsm_forward_field(chi, voxel_size=(1.0, 1.0, 1.0), b0_dir=(0.0, 0.0, 1.0),
                      pad_mm: float = 0.0) -> np.ndarray:
    """Field (ppm) = IFFT(D * FFT(chi)); ``pad_mm`` zero-pads to limit wrap-around."""
    pad = _pad_width(np.shape(chi), voxel_size, pad_mm)
    x = _pad(np.asarray(chi, dtype=np.float32), pad)
    d = qsm_dipole_kernel(x.shape, voxel_size, b0_dir)
    return _unpad(_ifftn_real(_fftn(x) * d), pad).astype(np.float32)


# ── I/O ─────────────────────────────────────────────────────────────────────

def _read_nifti_minimal(path):
    """NIfTI-1 reader (no nibabel): returns data (float64), affine, zooms."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rb") as fh:
        raw = fh.read()
    endian = "<" if int.from_bytes(raw[:4], "little") == 348 else ">"
    hdr = np.frombuffer(raw[:348], dtype=np.uint8)
    dim = np.frombuffer(raw[40:56], dtype=endian + "i2")
    datatype = int(np.frombuffer(raw[70:72], dtype=endian + "i2")[0])
    pixdim = np.frombuffer(raw[76:108], dtype=endian + "f4")
    vox_offset = int(np.frombuffer(raw[108:112], dtype=endian + "f4")[0])
    slope, inter = np.frombuffer(raw[112:120], dtype=endian + "f4")
    sform_code = int(np.frombuffer(raw[254:256], dtype=endian + "i2")[0])
    srow = np.frombuffer(raw[280:328], dtype=endian + "f4").reshape(3, 4)
    dt = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}[datatype]
    nd = int(dim[0])
    shape = tuple(int(s) for s in dim[1:1 + nd])
    data = np.frombuffer(raw[vox_offset:], dtype=endian + dt, count=int(np.prod(shape)))
    data = data.reshape(shape, order="F").astype(np.float64)
    if slope not in (0.0, 1.0) and np.isfinite(slope):
        data = data * slope + inter
    zooms = tuple(float(p) for p in pixdim[1:4])
    if sform_code > 0:
        aff = np.vstack([srow, [0, 0, 0, 1]]).astype(float)
    else:
        aff = np.diag(list(zooms) + [1.0])
    del hdr
    return data, aff, zooms


def _read_nifti(path):
    try:
        import nibabel as nib
    except ImportError:
        return _read_nifti_minimal(path)
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(dtype=np.float64)), img.affine, tuple(float(z) for z in img.header.get_zooms()[:3])


@register_function(
    aliases=["保存 NIfTI", "save nifti", "write QSM map", "write chi map nifti"],
    category="utils",
    description="Write a 3D map to NIfTI-1 (.nii or .nii.gz) with a given affine (voxel size taken "
                "from it or given), float32; uses nibabel when present, else a built-in writer.",
    requires={}, produces={}, auto_fix="none",
    examples=["ov.tl.qsm_save_nifti(res['chi'], 'out/sub-1_Chimap.nii', affine=data['affine'])"],
)
def qsm_save_nifti(data, path, affine=None, voxel_size=None) -> str:
    """Save ``data`` as float32 NIfTI-1 with the input's geometry; returns the path."""
    arr = np.asarray(data, dtype=np.float32)
    if affine is None:
        vs = _vox3(voxel_size if voxel_size is not None else 1.0)
        affine = np.diag(list(vs) + [1.0])
    affine = np.asarray(affine, dtype=float)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    try:
        import nibabel as nib
        img = nib.Nifti1Image(arr, affine)
        if voxel_size is not None:
            img.header.set_zooms(_vox3(voxel_size)[: arr.ndim] if arr.ndim == 3 else img.header.get_zooms())
        img.header.set_qform(affine, code=1)
        img.header.set_sform(affine, code=1)
        nib.save(img, str(path))
        return str(path)
    except ImportError:
        pass
    zooms = _vox3(voxel_size) if voxel_size is not None else tuple(np.linalg.norm(affine[:3, :3], axis=0))
    hdr = bytearray(352)
    hdr[0:4] = np.int32(348).tobytes()
    dims = np.ones(8, dtype=np.int16)
    dims[0] = arr.ndim
    dims[1:1 + arr.ndim] = arr.shape
    hdr[40:56] = dims.tobytes()
    hdr[70:72] = np.int16(16).tobytes()       # float32
    hdr[72:74] = np.int16(32).tobytes()       # bitpix
    pix = np.ones(8, dtype=np.float32)
    pix[1:4] = zooms
    hdr[76:108] = pix.tobytes()
    hdr[108:112] = np.float32(352).tobytes()  # vox_offset
    hdr[112:116] = np.float32(1).tobytes()    # scl_slope
    hdr[123] = 10                             # xyzt_units: mm, s
    hdr[254:256] = np.int16(1).tobytes()      # sform_code
    hdr[280:328] = np.asarray(affine[:3, :4], dtype=np.float32).tobytes()
    hdr[344:348] = b"n+1\x00"
    payload = bytes(hdr) + arr.tobytes(order="F")
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wb") as fh:
        fh.write(payload)
    return str(path)


def _scale_phase(p: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(p)), float(np.nanmax(p))
    if lo >= -np.pi - 1e-3 and hi <= np.pi + 1e-3:
        return p
    if lo >= -1e-3 and hi <= 2 * np.pi + 1e-3:
        return np.angle(np.exp(1j * p))
    # integer-coded phase (e.g. -4096..4095): map the observed range to [-pi, pi)
    warnings.warn(f"phase range [{lo:g}, {hi:g}] is not radians; rescaling to [-pi, pi)")
    return (p - lo) / (hi - lo) * 2 * np.pi - np.pi


@register_function(
    aliases=["读取多回波 GRE", "load multi-echo phase magnitude", "BIDS MEGRE loader",
             "QSM input loader", "多回波相位幅值读取"],
    category="utils",
    description="Load multi-echo GRE phase + magnitude NIfTI files (BIDS *_echo-<n>_part-phase/mag*) "
                "with their JSON sidecars: arrays (echo, x, y, z), echo times in seconds, field "
                "strength (T), B0 direction, voxel size, affine. Phase rescaled to radians if needed.",
    requires={}, produces={}, auto_fix="none",
    examples=["d = ov.tl.qsm_load_multi_echo('/data/sub-1/anat')",
              "d = ov.tl.qsm_load_multi_echo(phase_files=[...], magnitude_files=[...], echo_times=[...])"],
)
def qsm_load_multi_echo(directory: Optional[str] = None, phase_files: Optional[Sequence[str]] = None,
                        magnitude_files: Optional[Sequence[str]] = None,
                        echo_times: Optional[Sequence[float]] = None,
                        b0_tesla: Optional[float] = None, b0_dir=None) -> Dict:
    """Read a multi-echo GRE series.

    Either give ``directory`` (files matched by ``echo-<n>`` and ``part-phase`` /
    ``part-mag`` or ``phase`` / ``mag`` in their names) or explicit file lists in
    echo order.  Echo times, field strength and B0 direction come from the JSON
    sidecars (``EchoTime``, ``MagneticFieldStrength``, ``B0_dir``) unless given.
    Missing B0 direction defaults to the scanner z axis expressed in voxel
    coordinates via the affine.
    """
    if phase_files is None:
        if directory is None:
            raise ValueError("give a directory or phase_files")
        files = sorted(glob.glob(os.path.join(directory, "*.nii*")))

        def _echo(f):
            m = re.search(r"echo-?(\d+)", os.path.basename(f), re.I)
            return int(m.group(1)) if m else 0
        phase_files = sorted([f for f in files if re.search(r"part-phase|_phase|phase\.", os.path.basename(f), re.I)], key=_echo)
        magnitude_files = sorted([f for f in files if re.search(r"part-mag|_mag|magnitude", os.path.basename(f), re.I)], key=_echo)
        if not phase_files:
            raise FileNotFoundError(f"no phase NIfTI files in {directory}")
    phase, mag, tes, meta = [], [], [], {}
    affine = zooms = None
    for i, pf in enumerate(phase_files):
        p, affine, zooms = _read_nifti(pf)
        phase.append(_scale_phase(p).astype(np.float32))
        side = re.sub(r"\.nii(\.gz)?$", ".json", pf)
        if os.path.exists(side):
            with open(side) as fh:
                meta = json.load(fh)
            if echo_times is None and "EchoTime" in meta:
                tes.append(float(meta["EchoTime"]))
    if magnitude_files:
        for mf in magnitude_files:
            mag.append(np.abs(_read_nifti(mf)[0]).astype(np.float32))
    else:
        warnings.warn("no magnitude images: using uniform weights")
        mag = [np.ones_like(phase[0]) for _ in phase]
    if echo_times is None:
        if len(tes) != len(phase):
            raise ValueError("echo times not found in sidecars; pass echo_times")
        echo_times = tes
    te = _echo_times_seconds(echo_times)
    if b0_tesla is None:
        b0_tesla = meta.get("MagneticFieldStrength")
    if b0_dir is None:
        b0_dir = meta.get("B0_dir")
    if b0_dir is None:
        # scanner z expressed in voxel axes: column directions of the affine
        r = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
        b0_dir = tuple(np.linalg.solve(r, [0.0, 0.0, 1.0]))
    return dict(phase=np.stack(phase), magnitude=np.stack(mag), echo_times=te,
                b0_tesla=None if b0_tesla is None else float(b0_tesla),
                b0_dir=tuple(_unit(b0_dir)), voxel_size=_vox3(zooms), affine=affine,
                phase_files=list(phase_files), magnitude_files=list(magnitude_files or []),
                sidecar=meta)


# ── masking ─────────────────────────────────────────────────────────────────

@register_function(
    aliases=["脑掩膜", "QSM 脑掩膜", "magnitude brain mask", "brain extraction from GRE magnitude"],
    category="utils",
    description="Brain mask from GRE magnitude: threshold (default half the Otsu level of the "
                "first-echo magnitude), morphological opening to cut thin links to scalp, largest "
                "connected component, closing and hole filling. Generous by design; refine with "
                "the phase-reliability mask of qsm_multi_echo_field.",
    requires={}, produces={}, auto_fix="none",
    examples=["mask = ov.tl.qsm_brain_mask(d['magnitude'], voxel_size=d['voxel_size'])"],
)
def qsm_brain_mask(magnitude, voxel_size=(1.0, 1.0, 1.0), threshold: Optional[float] = None,
                   threshold_scale: float = 0.5, open_radius_mm: float = 4.0,
                   close_radius_mm: float = 2.0) -> np.ndarray:
    """Return a boolean brain mask.

    ``magnitude`` is 3D or (echo, x, y, z) (first echo used: highest SNR and
    least T2* dropout).  ``threshold`` overrides ``threshold_scale * Otsu``.
    A larger ``open_radius_mm`` separates brain from scalp more aggressively.
    """
    m = np.asarray(magnitude, dtype=np.float32)
    if m.ndim == 4:
        m = m[0]
    if threshold is None:
        threshold = threshold_scale * _otsu(m[m > 0])
    b = m > threshold
    core = ndi.binary_erosion(b, _ball(open_radius_mm, voxel_size)) if open_radius_mm > 0 else b
    if not core.any():
        raise ValueError("mask is empty after opening; lower open_radius_mm or threshold")
    core = _largest_component(core)
    out = (ndi.binary_dilation(core, _ball(open_radius_mm, voxel_size)) & b) if open_radius_mm > 0 else core
    if close_radius_mm > 0:
        out = ndi.binary_closing(out, _ball(close_radius_mm, voxel_size))
    return ndi.binary_fill_holes(out)


# ── phase unwrapping ────────────────────────────────────────────────────────

def _laplacian_eig(shape, voxel_size):
    vs = _vox3(voxel_size)
    out = np.zeros(shape, dtype=np.float64)
    for ax, (n, d) in enumerate(zip(shape, vs)):
        k = np.arange(n)
        v = (2 * np.cos(2 * np.pi * k / n) - 2) / d ** 2
        s = [1, 1, 1]
        s[ax] = n
        out = out + v.reshape(s)
    return out


def _laplacian_unwrap(phase, voxel_size):
    lam = _laplacian_eig(phase.shape, voxel_size)
    c, s = np.cos(phase), np.sin(phase)
    lap = lambda x: _ifftn_real(_fftn(x) * lam)
    lp = c * lap(s) - s * lap(c)
    inv = np.zeros_like(lam)
    nz = lam != 0
    inv[nz] = 1.0 / lam[nz]
    return _ifftn_real(_fftn(lp) * inv)


@register_function(
    aliases=["相位解缠", "3D phase unwrapping", "quality-guided unwrapping", "Laplacian unwrapping"],
    category="utils",
    description="3D spatial phase unwrapping inside a mask: 'quality' = reliability-sorted path "
                "following (Herraez 2002, exact 2*pi multiples, scikit-image) or 'laplacian' "
                "(Schofield & Zhu 2003; smooth, robust to noise, not integer-exact).",
    requires={}, produces={}, auto_fix="none",
    examples=["u = ov.tl.qsm_unwrap_phase(phase_echo, mask, method='quality')"],
)
def qsm_unwrap_phase(phase, mask=None, method: str = "quality", voxel_size=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Unwrap a 3D wrapped phase (radians). Outside ``mask`` the output is 0."""
    p = np.asarray(phase, dtype=np.float64)
    m = np.ones(p.shape, bool) if mask is None else np.asarray(mask, bool)
    if method == "quality":
        try:
            from skimage.restoration import unwrap_phase
        except ImportError:
            warnings.warn("scikit-image missing: falling back to Laplacian unwrapping")
            method = "laplacian"
        else:
            u = unwrap_phase(np.ma.array(p, mask=~m))
            return np.ma.filled(u, 0.0).astype(np.float32)
    if method == "laplacian":
        u = _laplacian_unwrap(np.where(m, p, 0.0), voxel_size)
        # snap to the nearest 2*pi-consistent value of the measured phase
        u = p + 2 * np.pi * np.round((u - p) / (2 * np.pi))
        return np.where(m, u, 0.0).astype(np.float32)
    raise ValueError("method must be 'quality' or 'laplacian'")


# ── multi-echo field estimation ─────────────────────────────────────────────

@register_function(
    aliases=["多回波场图", "B0 场图估计", "multi-echo field map", "frequency map from phase",
             "MCPC-3D-S phase offset", "echo combination QSM"],
    category="utils",
    description="Total field (Hz) from multi-echo GRE phase: MCPC-3D-S phase-offset removal, "
                "per-echo 3D unwrapping, 2*pi alignment of echoes, magnitude^2-weighted LS fit "
                "phase = 2*pi*f*TE through the origin; returns field, weighted fit residual (rad) and "
                "a phase-reliability mask.",
    requires={}, produces={}, auto_fix="none",
    examples=["fe = ov.tl.qsm_multi_echo_field(d['phase'], d['magnitude'], d['echo_times'], mask)"],
)
def qsm_multi_echo_field(phase, magnitude, echo_times, mask, unwrap: str = "quality",
                         offset_sigma_mm: float = 2.0, max_residual: float = 0.5,
                         voxel_size=(1.0, 1.0, 1.0), remove_offset: bool = True) -> Dict:
    """Estimate the total field in Hz.

    phase, magnitude : (n_echo, x, y, z); phase in radians.
    echo_times : seconds (ms are detected and converted).
    max_residual : voxels whose magnitude-weighted RMS residual of the linear
        phase-vs-TE fit exceeds this (radians) are unreliable (noise, signal
        voids, unwrapping failures) and are left out of ``reliable_mask``.
    """
    ph = np.asarray(phase, dtype=np.float32)
    mg = np.asarray(magnitude, dtype=np.float32)
    if ph.ndim == 3:
        ph, mg = ph[None], mg[None]
    te = _echo_times_seconds(echo_times)
    if len(te) != ph.shape[0]:
        raise ValueError(f"{len(te)} echo times for {ph.shape[0]} echoes")
    m = np.asarray(mask, bool)
    order = np.argsort(te)
    ph, mg, te = ph[order], mg[order], te[order]
    n = len(te)
    z = mg * np.exp(1j * ph)
    offset = np.zeros(m.shape, np.float32)
    guide = None
    if n >= 2:
        dte = te[1] - te[0]
        du = qsm_unwrap_phase(np.angle(z[1] * np.conj(z[0])), m, unwrap, voxel_size).astype(np.float64)
        guide = du / dte                                     # rad/s
        if remove_offset:
            off = np.angle(z[0] * np.exp(-1j * guide * te[0]))
            sig = [offset_sigma_mm / v for v in _vox3(voxel_size)]
            # smooth only inside the mask: outside it the guide is undefined
            oc = np.where(m, mg[0], 0) * np.exp(1j * off)
            sm = ndi.gaussian_filter(oc.real, sig) + 1j * ndi.gaussian_filter(oc.imag, sig)
            offset = np.angle(sm).astype(np.float32)
            z = z * np.exp(-1j * offset)[None]
    elif remove_offset:
        warnings.warn("single echo: the phase offset cannot be separated from the field")
    pc = np.angle(z)
    U = np.empty(ph.shape, np.float32)
    for e in range(n):
        u = qsm_unwrap_phase(pc[e], m, unwrap, voxel_size)
        ref = guide * te[e] if guide is not None else (U[0] * te[e] / te[0] if e else None)
        if ref is not None:
            k = np.round(np.median((u - ref)[m]) / (2 * np.pi))
            u = u - np.float32(2 * np.pi * k)
        U[e] = u
    w = mg ** 2
    T = te[:, None, None, None].astype(np.float32)
    num = (w * T * U).sum(0)
    den = (w * T ** 2).sum(0)
    omega = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    field = (omega / (2 * np.pi)).astype(np.float32)
    field[~m] = 0
    if n >= 2:
        res = np.sqrt((w * (U - omega[None] * T) ** 2).sum(0) / np.maximum(w.sum(0), 1e-12)).astype(np.float32)
    else:
        res = np.zeros(m.shape, np.float32)
    res[~m] = 0
    rel = m & (res < max_residual)
    rel = ndi.binary_fill_holes(ndi.binary_opening(rel))
    rel = _largest_component(rel) if rel.any() else rel
    return dict(field_hz=field, residual=res, reliable_mask=rel, phase_offset=offset,
                unwrapped=U, echo_times=te)


# ── background field removal ────────────────────────────────────────────────

def _smv_k(shape, voxel_size, radius_mm):
    vs = _vox3(voxel_size)
    g = np.meshgrid(*[(sfft.fftfreq(n) * n) * v for n, v in zip(shape, vs)], indexing="ij", sparse=True)
    s = ((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) <= radius_mm ** 2).astype(np.float64)
    s /= s.sum()
    return _fftn(s).real


def _vsharp(field, mask, voxel_size, radii, threshold):
    F = _fftn(field)
    M = _fftn(mask.astype(np.float64))
    out = np.zeros(field.shape)
    done = np.zeros(field.shape, bool)
    inv = None
    delta = 1 - np.sqrt(1e-7)
    for r in sorted(radii, reverse=True):
        S = _smv_k(field.shape, voxel_size, r)
        if inv is None:
            om = 1 - S
            inv = np.where(np.abs(om) < threshold, 0.0, 1.0 / np.where(np.abs(om) < threshold, 1, om))
        cur = _ifftn_real(M * S) > delta
        hp = _ifftn_real(F * (1 - S))
        sel = cur & ~done
        out[sel] = hp[sel]
        done |= sel
    loc = _ifftn_real(_fftn(out) * inv)
    loc[~done] = 0
    return loc, done


def _pdf(field, mask, voxel_size, b0_dir, tol, max_iter, pad):
    from scipy.sparse.linalg import LinearOperator, cg
    padw = pad
    f = _pad(field, padw)
    m = _pad(mask, padw)
    d = qsm_dipole_kernel(f.shape, voxel_size, b0_dir).astype(np.float64)
    out = ~m
    n = int(out.sum())
    shp = f.shape

    def A(v):
        x = np.zeros(shp)
        x[out] = v
        y = _ifftn_real(_fftn(x) * d) * m
        return _ifftn_real(_fftn(y) * d)[out]

    b = _ifftn_real(_fftn(f * m) * d)[out]
    op = LinearOperator((n, n), matvec=A, dtype=np.float64)
    try:
        xb, _ = cg(op, b, rtol=tol, maxiter=max_iter)
    except TypeError:            # scipy < 1.12
        xb, _ = cg(op, b, tol=tol, maxiter=max_iter)
    x = np.zeros(shp)
    x[out] = xb
    bg = _ifftn_real(_fftn(x) * d)
    loc = (f - bg) * m
    return _unpad(loc, padw), _unpad(m, padw)


@register_function(
    aliases=["背景场去除", "background field removal", "V-SHARP", "SHARP", "PDF background field",
             "local field QSM"],
    category="utils",
    description="Remove the background field (sources outside the mask) from a total field: "
                "V-SHARP (multi-radius SMV, QSM.jl default radii 18*vox..2*vox step 2*vox, "
                "threshold 0.05), SHARP (single radius) or PDF (projection onto dipole fields, CG). "
                "Returns the local field (same units as input) and the mask it is valid in.",
    requires={}, produces={}, auto_fix="none",
    examples=["loc, lmask = ov.tl.qsm_background_removal(field_hz, mask, voxel_size, method='vsharp')"],
)
def qsm_background_removal(field, mask, voxel_size=(1.0, 1.0, 1.0), method: str = "vsharp",
                           radii_mm: Optional[Sequence[float]] = None, threshold: float = 0.05,
                           b0_dir=(0.0, 0.0, 1.0), tol: float = 1e-5, max_iter: int = 200,
                           pad_mm: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(local_field, local_mask)``.

    V-SHARP / SHARP erode the mask by the smallest radius (edge voxels are lost
    and set to 0); PDF keeps the whole mask but needs ``b0_dir``.
    """
    f = np.asarray(field, dtype=np.float64)
    m = np.asarray(mask, bool)
    f = np.where(m, f, 0.0)
    vs = _vox3(voxel_size)
    if method in ("vsharp", "sharp"):
        if radii_mm is None:
            hi, lo = 18 * min(vs), 2 * max(vs)
            radii_mm = list(np.arange(hi, lo - 1e-9, -2 * max(vs))) if method == "vsharp" else [6 * max(vs)]
        loc, lm = _vsharp(f, m, vs, list(radii_mm), threshold)
    elif method == "pdf":
        loc, lm = _pdf(f, m, vs, b0_dir, tol, max_iter, _pad_width(f.shape, vs, pad_mm))
    else:
        raise ValueError("method must be 'vsharp', 'sharp' or 'pdf'")
    return loc.astype(np.float32), lm


# ── dipole inversion ────────────────────────────────────────────────────────

def _diff_k(shape):
    E = []
    for ax, n in enumerate(shape):
        k = sfft.fftfreq(n)
        s = [1, 1, 1]
        s[ax] = n
        E.append((np.exp(2j * np.pi * k) - 1).reshape(s))
    return E


def _grad(x):
    return [np.roll(x, -1, axis=a) - x for a in range(3)]


def _grad_adj(g):
    return sum(np.roll(g[a], 1, axis=a) - g[a] for a in range(3))


def _tv_admm(phi, d, lam, rho, n_iter, tol):
    # single precision keeps the memory of a whole-head volume within ~1 GB
    phi = phi.astype(np.float32)
    E = _diff_k(phi.shape)
    ete = sum(np.abs(e) ** 2 for e in E)
    den = (d.astype(np.float64) ** 2 + rho * ete).astype(np.float32)
    den[den == 0] = 1.0
    dphi = (d * _fftn(phi)).astype(np.complex64)
    x = np.zeros(phi.shape, np.float32)
    zs = [np.zeros(phi.shape, np.float32) for _ in range(3)]
    us = [np.zeros(phi.shape, np.float32) for _ in range(3)]
    thr = lam / rho
    for _ in range(n_iter):
        x_old = x
        x = _ifftn_real((dphi + rho * _fftn(_grad_adj([zs[a] - us[a] for a in range(3)]))) / den)
        g = _grad(x)
        for a in range(3):
            v = g[a] + us[a]
            zs[a] = (np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)).astype(np.float32)
            us[a] = us[a] + g[a] - zs[a]
        dx = np.linalg.norm(x - x_old) / max(np.linalg.norm(x), 1e-12)
        if dx < tol:
            break
    return x


@register_function(
    aliases=["偶极子反演", "dipole inversion", "QSM inversion", "TV QSM", "Tikhonov QSM", "TKD",
             "susceptibility from local field"],
    category="utils",
    description="Susceptibility (same units as the local field, use ppm) from a local field: "
                "'tv' = total-variation ADMM (Bilgic 2014; least streaking, contrast closest to 1:1), "
                "'tikhonov' (closed form, identity or gradient penalty), 'tkd' (truncated k-space "
                "division). Output is zero outside the mask.",
    requires={}, produces={}, auto_fix="none",
    examples=["chi = ov.tl.qsm_dipole_inversion(local_ppm, lmask, voxel_size, b0_dir, method='tv')"],
)
def qsm_dipole_inversion(local_field, mask, voxel_size=(1.0, 1.0, 1.0), b0_dir=(0.0, 0.0, 1.0),
                         method: str = "tv", lam: Optional[float] = None, rho: Optional[float] = None,
                         n_iter: int = 50, tol: float = 1e-3, regularization: str = "identity",
                         tkd_threshold: float = 0.15, pad_mm: float = 0.0) -> np.ndarray:
    """Invert the dipole convolution.

    Defaults (field in ppm, ~1 mm voxels): tv lam=3e-4 (rho=100*lam);
    tikhonov lam=1e-2 (identity) / 1e-2 (gradient).  Larger lam -> smoother,
    lower contrast; smaller -> more noise and streaking.  Check the choice with
    the dipole-consistency residual and on a numerical phantom.
    """
    m = np.asarray(mask, bool)
    vs = _vox3(voxel_size)
    pad = _pad_width(m.shape, vs, pad_mm)
    phi = _pad(np.where(m, np.asarray(local_field, dtype=np.float64), 0.0), pad)
    d = qsm_dipole_kernel(phi.shape, vs, b0_dir).astype(np.float64)
    if method == "tv":
        lam = 3e-4 if lam is None else lam
        rho = 100 * lam if rho is None else rho
        chi = _tv_admm(phi, d, lam, rho, n_iter, tol).astype(np.float64)
    elif method == "tikhonov":
        lam = 1e-2 if lam is None else lam
        if regularization == "identity":
            g = 1.0
        elif regularization == "gradient":
            g = sum(np.abs(e) ** 2 for e in _diff_k(phi.shape))
        else:
            raise ValueError("regularization must be 'identity' or 'gradient'")
        den = d ** 2 + lam * g
        den = np.where(den == 0, 1.0, den)
        chi = _ifftn_real(_fftn(phi) * d / den)
    elif method == "tkd":
        inv = np.where(np.abs(d) > tkd_threshold, 1.0 / np.where(np.abs(d) > tkd_threshold, d, 1), 0.0)
        chi = _ifftn_real(_fftn(phi) * inv)
    else:
        raise ValueError("method must be 'tv', 'tikhonov' or 'tkd'")
    chi = _unpad(chi, pad)
    chi[~m] = 0
    return chi.astype(np.float32)


# ── full pipeline ───────────────────────────────────────────────────────────

@register_function(
    aliases=["QSM 重建", "定量磁化率成像", "quantitative susceptibility mapping",
             "QSM pipeline", "chi map from multi-echo GRE", "susceptibility map reconstruction"],
    category="utils",
    description="End-to-end QSM from multi-echo GRE phase + magnitude: brain mask, MCPC-3D-S offset "
                "removal + quality-guided unwrapping + weighted echo fit (Hz), phase-reliability "
                "mask refinement, background removal (V-SHARP / PDF), Hz->ppm with gamma_bar*B0, "
                "dipole inversion (TV / Tikhonov / TKD). Returns chi (ppm) and every intermediate "
                "with a QC summary.",
    requires={}, produces={}, auto_fix="none",
    examples=[
        "d = ov.tl.qsm_load_multi_echo('/data/sub-1/anat')",
        "res = ov.tl.qsm_reconstruct(d['phase'], d['magnitude'], d['echo_times'], d['b0_tesla'], "
        "d['voxel_size'], b0_dir=d['b0_dir'])",
        "ov.tl.qsm_save_nifti(res['chi'], 'derivatives/qsm/sub-1_Chimap.nii', affine=d['affine'])",
    ],
)
def qsm_reconstruct(phase, magnitude, echo_times, b0_tesla: float, voxel_size=(1.0, 1.0, 1.0),
                    b0_dir=(0.0, 0.0, 1.0), mask=None, refine_mask: bool = True,
                    unwrap: str = "quality", background: str = "vsharp", inversion: str = "tv",
                    lam: Optional[float] = None, max_residual: float = 0.5,
                    reference: Optional[str] = None, gamma_bar_mhz_per_t: float = GAMMA_BAR_MHZ_PER_T,
                    **inversion_kw) -> Dict:
    """Run the full chain and return a dict.

    Keys: chi (ppm, 0 outside local_mask), field_hz, local_field_ppm, mask,
    reliable_mask, local_mask, residual, summary.  ``reference='mask_mean'``
    subtracts the mean chi inside the local mask (QSM is relative; leave None
    unless a reference region / convention is required).
    """
    if b0_tesla is None or not np.isfinite(b0_tesla) or b0_tesla <= 0:
        raise ValueError("b0_tesla (field strength in T) is required")
    vs = _vox3(voxel_size)
    ph = np.asarray(phase, dtype=np.float32)
    mg = np.asarray(magnitude, dtype=np.float32)
    if ph.ndim == 3:
        ph, mg = ph[None], mg[None]
    m0 = qsm_brain_mask(mg, vs) if mask is None else np.asarray(mask, bool)
    fe = qsm_multi_echo_field(ph, mg, echo_times, m0, unwrap=unwrap, max_residual=max_residual,
                              voxel_size=vs)
    m1 = fe["reliable_mask"] if refine_mask else m0
    loc_hz, lm = qsm_background_removal(np.where(m1, fe["field_hz"], 0), m1, vs, method=background,
                                        b0_dir=b0_dir)
    scale = 1.0 / (gamma_bar_mhz_per_t * float(b0_tesla))   # Hz -> ppm
    loc_ppm = (loc_hz * scale).astype(np.float32)
    chi = qsm_dipole_inversion(loc_ppm, lm, vs, b0_dir, method=inversion, lam=lam, **inversion_kw)
    if reference == "mask_mean":
        chi = np.where(lm, chi - chi[lm].mean(), 0).astype(np.float32)
    pred = qsm_forward_field(chi, vs, b0_dir)
    r = loc_ppm[lm] - loc_ppm[lm].mean()
    consistency = float(np.linalg.norm((pred[lm] - pred[lm].mean()) - r) / max(np.linalg.norm(r), 1e-12))
    summary = dict(
        n_echoes=int(ph.shape[0]), echo_times_s=[float(t) for t in fe["echo_times"]],
        b0_tesla=float(b0_tesla), b0_dir=tuple(_unit(b0_dir)), voxel_size=vs,
        mask_voxels=int(m0.sum()), reliable_voxels=int(m1.sum()), local_voxels=int(lm.sum()),
        residual_p50_p99=[float(np.percentile(fe["residual"][m0], q)) for q in (50, 99)],
        field_hz_p1_p50_p99=[float(np.percentile(fe["field_hz"][m1], q)) for q in (1, 50, 99)],
        local_ppm_p1_p99=[float(np.percentile(loc_ppm[lm], q)) for q in (1, 99)],
        chi_ppm_p1_p50_p99=[float(np.percentile(chi[lm], q)) for q in (1, 50, 99)],
        dipole_inconsistency=consistency, background=background, inversion=inversion,
    )
    return dict(chi=chi, field_hz=fe["field_hz"], local_field_ppm=loc_ppm, mask=m0,
                reliable_mask=m1, local_mask=lm, residual=fe["residual"],
                phase_offset=fe["phase_offset"], summary=summary)


# ── numerical phantom ───────────────────────────────────────────────────────

@register_function(
    aliases=["QSM 数值体模", "numerical susceptibility phantom", "QSM phantom",
             "simulate multi-echo GRE"],
    category="utils",
    description="Numerical head phantom with known susceptibility for validating a QSM pipeline: "
                "ellipsoidal brain with nuclei of different chi, a vessel and a strongly negative "
                "inclusion, surrounded by a low-signal shell and air-like external sources; "
                "simulates wrapped multi-echo phase (with phase offset) and T2*-decaying magnitude.",
    requires={}, produces={}, auto_fix="none",
    examples=["ph = ov.tl.qsm_numerical_phantom((96, 96, 80), b0_tesla=3.0, echo_times=[0.005, 0.01, 0.015, 0.02])"],
)
def qsm_numerical_phantom(shape=(80, 80, 64), voxel_size=(1.0, 1.0, 1.0), b0_tesla: float = 3.0,
                          echo_times=(0.004, 0.008, 0.012, 0.016), b0_dir=(0.0, 0.0, 1.0),
                          noise: float = 0.01, seed: int = 0) -> Dict:
    """Return dict(phase, magnitude, chi, brain_mask, structures, field_hz, ...).

    All values are generic textbook magnitudes chosen to exercise the steps
    (wraps, background field, positive / negative sources); they are not
    calibrated to any dataset.
    """
    rng = np.random.default_rng(seed)
    vs = _vox3(voxel_size)
    shp = tuple(int(s) for s in shape)
    g = np.meshgrid(*[(np.arange(n) - (n - 1) / 2) * v for n, v in zip(shp, vs)], indexing="ij")
    ext = [n * v / 2 for n, v in zip(shp, vs)]

    def ell(c, r):
        return sum(((g[i] - c[i] * ext[i]) / (r[i] * ext[i])) ** 2 for i in range(3)) <= 1

    head = ell((0, 0, 0), (0.85, 0.85, 0.85))
    brain = ell((0, 0, 0), (0.68, 0.68, 0.68))
    chi = np.zeros(shp, np.float32)
    structures = {}
    for name, c, r, val in [("nucleus_a", (0.25, 0.2, 0.0), (0.12, 0.1, 0.12), 0.12),
                            ("nucleus_b", (-0.25, 0.2, 0.0), (0.1, 0.12, 0.1), 0.07),
                            ("nucleus_c", (0.0, -0.3, 0.1), (0.1, 0.1, 0.12), 0.04),
                            ("nucleus_d", (0.0, 0.3, -0.25), (0.08, 0.08, 0.08), -0.03)]:
        s = ell(c, r) & brain
        chi[s] = val
        structures[name] = (s, val)
    vessel = (((g[0] - 0.35 * ext[0]) ** 2 + (g[1] + 0.1 * ext[1]) ** 2) <= (1.5 * vs[0]) ** 2) & brain
    chi[vessel] = 0.3
    structures["vessel"] = (vessel, 0.3)
    inclusion = ell((-0.3, -0.25, -0.2), (0.04, 0.04, 0.04)) & brain
    chi[inclusion] = -1.5
    structures["negative_inclusion"] = (inclusion, -1.5)
    # air-like external sources (sinus / ear canals) and the head-air boundary
    full = chi.copy()
    full[~head] = 9.0
    for c in [(0.0, 0.75, -0.6), (0.7, 0.0, -0.4), (-0.7, 0.0, -0.4)]:
        full[ell(c, (0.15, 0.12, 0.12)) & head & ~brain] = 9.0
    total_ppm = qsm_forward_field(full, vs, b0_dir, pad_mm=0.25 * max(ext) * 2)
    field_hz = (total_ppm * GAMMA_BAR_MHZ_PER_T * b0_tesla).astype(np.float32)
    field_hz[~head] = 0
    te = _echo_times_seconds(echo_times)
    offset = (0.8 * np.sin(g[0] / ext[0] * 2) + 0.5 * np.cos(g[1] / ext[1] * 1.5)).astype(np.float32)
    rho = np.where(brain, 1.0, np.where(head, 0.25, 0.0)).astype(np.float32)
    rho[vessel] = 0.6
    r2s = np.where(brain, 30.0, 60.0).astype(np.float32)
    ph, mg = [], []
    for t in te:
        mag = rho * np.exp(-r2s * t)
        sig = mag * np.exp(1j * (2 * np.pi * field_hz * t + offset))
        sig = sig + noise * (rng.standard_normal(shp) + 1j * rng.standard_normal(shp))
        ph.append(np.angle(sig).astype(np.float32))
        mg.append(np.abs(sig).astype(np.float32))
    return dict(phase=np.stack(ph), magnitude=np.stack(mg), echo_times=te, chi=chi,
                brain_mask=brain, head_mask=head, structures=structures, field_hz=field_hz,
                phase_offset=offset, b0_tesla=float(b0_tesla), b0_dir=tuple(_unit(b0_dir)),
                voxel_size=vs)
