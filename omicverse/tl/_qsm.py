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


def _rshape(shape):
    return tuple(shape[:-1]) + (shape[-1] // 2 + 1,)


def _rfft(x):
    """Real-input FFT (half spectrum); float32 in -> complex64 out."""
    return sfft.rfftn(x, workers=_WORKERS)


def _irfft(X, shape):
    return sfft.irfftn(X, s=shape, workers=_WORKERS)


def _rkgrid(shape, voxel_size):
    vs = _vox3(voxel_size)
    ks = [sfft.fftfreq(shape[0], d=vs[0]), sfft.fftfreq(shape[1], d=vs[1]),
          sfft.rfftfreq(shape[2], d=vs[2])]
    return np.meshgrid(*ks, indexing="ij", sparse=True)


def _dipole_from_grid(kx, ky, kz, b0_dir):
    b = _unit(b0_dir)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    kb = kx * b[0] + ky * b[1] + kz * b[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1.0 / 3.0 - kb ** 2 / k2
    d[k2 == 0] = 0.0
    return d.astype(np.float32)


def _dipole_r(shape, voxel_size, b0_dir):
    """Dipole kernel on the half (rfft) spectrum; D(k) = D(-k), so this is exact."""
    return _dipole_from_grid(*_rkgrid(shape, voxel_size), b0_dir)


def _bbox(mask, margin=0):
    """Slices of the mask's bounding box grown by ``margin`` voxels (clipped)."""
    sl = []
    for ax in range(mask.ndim):
        idx = np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != ax)))
        lo = max(int(idx[0]) - margin, 0) if idx.size else 0
        hi = min(int(idx[-1]) + 1 + margin, mask.shape[ax]) if idx.size else mask.shape[ax]
        sl.append(slice(lo, hi))
    return tuple(sl)


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
    return _dipole_from_grid(*_kgrid(shape, voxel_size), b0_dir)


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
    shape = x.shape
    X = _rfft(x)
    del x
    X *= _dipole_r(shape, voxel_size, b0_dir)
    return np.ascontiguousarray(_unpad(_irfft(X, shape), pad), dtype=np.float32)


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
    phase, mag, tes, meta = None, None, [], {}
    affine = zooms = None
    for i, pf in enumerate(phase_files):
        p, affine, zooms = _read_nifti(pf)
        if phase is None:
            phase = np.empty((len(phase_files),) + p.shape, np.float32)
        phase[i] = _scale_phase(p)
        del p
        side = re.sub(r"\.nii(\.gz)?$", ".json", pf)
        if os.path.exists(side):
            with open(side) as fh:
                meta = json.load(fh)
            if echo_times is None and "EchoTime" in meta:
                tes.append(float(meta["EchoTime"]))
    if magnitude_files:
        mag = np.empty((len(magnitude_files),) + phase.shape[1:], np.float32)
        for i, mf in enumerate(magnitude_files):
            mag[i] = np.abs(_read_nifti(mf)[0])
    else:
        warnings.warn("no magnitude images: using uniform weights")
        mag = np.ones_like(phase)
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
    return dict(phase=phase, magnitude=mag, echo_times=te,
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

def _laplacian_eig_r(shape, voxel_size):
    """Eigenvalues of the 7-point Laplacian on the half (rfft) spectrum, float32."""
    vs = _vox3(voxel_size)
    out = np.zeros(_rshape(shape), dtype=np.float32)
    for ax, (n, d) in enumerate(zip(shape, vs)):
        k = np.arange(n // 2 + 1 if ax == 2 else n)
        v = ((2 * np.cos(2 * np.pi * k / n) - 2) / d ** 2).astype(np.float32)
        s = [1, 1, 1]
        s[ax] = v.size
        out += v.reshape(s)
    return out


def _laplacian_unwrap(phase, voxel_size):
    shape = phase.shape
    lam = _laplacian_eig_r(shape, voxel_size)
    c, s = np.cos(phase), np.sin(phase)
    lp = c * _irfft(_rfft(s) * lam, shape)
    lp -= s * _irfft(_rfft(c) * lam, shape)
    del c, s
    nz = lam != 0
    inv = np.zeros_like(lam)
    inv[nz] = 1.0 / lam[nz]
    return _irfft(_rfft(lp) * inv, shape)


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
    """Unwrap a 3D wrapped phase (radians). Outside ``mask`` the output is 0.

    Works on the mask's bounding box only (memory and time scale with the
    object, not the field of view).
    """
    p = np.asarray(phase, dtype=np.float32)
    m = np.ones(p.shape, bool) if mask is None else np.asarray(mask, bool)
    out = np.zeros(p.shape, np.float32)
    if not m.any():
        return out
    if method == "quality":
        try:
            from skimage.restoration import unwrap_phase
        except ImportError:
            warnings.warn("scikit-image missing: falling back to Laplacian unwrapping")
            method = "laplacian"
        else:
            sl = _bbox(m, 1)
            mc = m[sl]
            u = unwrap_phase(np.ma.array(p[sl].astype(np.float64), mask=~mc))
            out[sl] = np.ma.filled(u, 0.0)
            return out
    if method == "laplacian":
        sl = _bbox(m, 8)
        pc, mc = p[sl], m[sl]
        u = _laplacian_unwrap(np.where(mc, pc, np.float32(0)), voxel_size)
        # snap to the nearest 2*pi-consistent value of the measured phase
        u = pc + np.float32(2 * np.pi) * np.round((u - pc) / np.float32(2 * np.pi))
        out[sl] = np.where(mc, u, 0)
        return out
    raise ValueError("method must be 'quality' or 'laplacian'")


# ── multi-echo field estimation ─────────────────────────────────────────────

def _wrap(x):
    """Wrap radians to [-pi, pi) in the input's precision."""
    return (x + np.float32(np.pi)) % np.float32(2 * np.pi) - np.float32(np.pi)


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
                         voxel_size=(1.0, 1.0, 1.0), remove_offset: bool = True,
                         echo_unwrap: str = "auto", return_unwrapped: bool = True) -> Dict:
    """Estimate the total field in Hz.

    phase, magnitude : (n_echo, x, y, z); phase in radians.
    echo_times : seconds (ms are detected and converted).
    echo_unwrap : 'spatial' (3D-unwrap every echo, align by whole cycles),
        'temporal' (cycle counts from the unwrapped echo-difference frequency)
        or 'auto' (per voxel, whichever fits phase = 2*pi*f*TE better).
    max_residual : voxels whose magnitude-weighted RMS residual of the linear
        phase-vs-TE fit exceeds this (radians) are unreliable (noise, signal
        voids, unwrapping failures) and are left out of ``reliable_mask``.
    return_unwrapped : also return the unwrapped echoes (n_echo x volume float32).

    Single precision, processed per echo inside the mask's bounding box.
    """
    ph_in, mg_in = phase, magnitude
    if np.ndim(ph_in) == 3:
        ph_in, mg_in = np.asarray(ph_in)[None], np.asarray(mg_in)[None]
    te = _echo_times_seconds(echo_times)
    if len(te) != len(ph_in):
        raise ValueError(f"{len(te)} echo times for {len(ph_in)} echoes")
    vs = _vox3(voxel_size)
    m_full = np.asarray(mask, bool)
    shape = m_full.shape
    sig = [offset_sigma_mm / v for v in vs]
    sl = _bbox(m_full, int(np.ceil(4 * max(sig))) + 1)
    m = m_full[sl]
    order = np.argsort(te)
    te = te[order]
    n = len(te)
    ph = [np.asarray(ph_in[i][sl], dtype=np.float32) for i in order]
    mg = [np.asarray(mg_in[i][sl], dtype=np.float32) for i in order]
    tef = [np.float32(t) for t in te]
    offset = np.zeros(m.shape, np.float32)
    guide = None
    if n >= 2:
        du = qsm_unwrap_phase(_wrap(ph[1] - ph[0]), m, unwrap, vs)
        guide = du / np.float32(te[1] - te[0])                # rad/s
        del du
        if remove_offset:
            off = _wrap(ph[0] - guide * tef[0])
            # smooth only inside the mask: outside it the guide is undefined
            a = np.where(m, mg[0], np.float32(0))
            re_ = ndi.gaussian_filter(a * np.cos(off), sig)
            im_ = ndi.gaussian_filter(a * np.sin(off), sig)
            offset = np.arctan2(im_, re_).astype(np.float32)
            del off, a, re_, im_
    elif remove_offset:
        warnings.warn("single echo: the phase offset cannot be separated from the field")
    pc = [_wrap(ph[e] - offset) for e in range(n)]
    del ph
    w = [g * g for g in mg]
    del mg
    den = sum(w[e] * tef[e] ** 2 for e in range(n))
    wsum = np.maximum(sum(w), np.float32(1e-12))

    def _fit(U):
        num = sum(w[e] * tef[e] * U[e] for e in range(n))
        om = np.where(den > 0, num / np.where(den > 0, den, 1), 0).astype(np.float32)
        del num
        r2 = sum(w[e] * (U[e] - om * tef[e]) ** 2 for e in range(n))
        return om, np.sqrt(r2 / wsum).astype(np.float32)

    # spatial: every echo unwrapped in 3D, echoes aligned by whole cycles
    U = []
    for e in range(n):
        u = qsm_unwrap_phase(pc[e], m, unwrap, vs)
        ref = guide * tef[e] if guide is not None else (U[0] * (tef[e] / tef[0]) if e else None)
        if ref is not None:
            k = np.round(np.median((u - ref)[m]) / (2 * np.pi))
            u -= np.float32(2 * np.pi * k)
        U.append(u)
    omega, res = _fit(U)
    if guide is not None and echo_unwrap in ("auto", "temporal"):
        # temporal: each echo's cycle count from the unwrapped echo-difference
        # frequency (fewer wraps over the short spacing); keeps thin structures
        # whose phase jump at late echoes exceeds pi (veins, calcifications)
        two_pi = np.float32(2 * np.pi)
        Ut = [pc[e] + two_pi * np.round((guide * tef[e] - pc[e]) / two_pi) for e in range(n)]
        om_t, res_t = _fit(Ut)
        pick = (res_t < res) if echo_unwrap == "auto" else np.ones(m.shape, bool)
        omega = np.where(pick, om_t, omega)
        res = np.where(pick, res_t, res)
        U = [np.where(pick, Ut[e], U[e]) for e in range(n)]
        del Ut, om_t, res_t
    del pc, w, den, wsum, guide
    if n < 2:
        res = np.zeros(m.shape, np.float32)
    f = (omega / np.float32(2 * np.pi)).astype(np.float32)
    f[~m] = 0
    res[~m] = 0
    rel = m & (res < max_residual)
    rel = ndi.binary_fill_holes(ndi.binary_opening(rel))
    rel = _largest_component(rel) if rel.any() else rel

    def _full(x, dtype=np.float32):
        o = np.zeros(shape, dtype)
        o[sl] = x
        return o
    out = dict(field_hz=_full(f), residual=_full(res), reliable_mask=_full(rel, bool),
               phase_offset=_full(offset), echo_times=te)
    if return_unwrapped:
        uw = np.zeros((n,) + shape, np.float32)
        for e in range(n):
            uw[e][sl] = np.where(m, U[e], 0)
        out["unwrapped"] = uw
    return out


# ── background field removal ────────────────────────────────────────────────

def _smv_kr(shape, voxel_size, radius_mm):
    """Half-spectrum (rfft) of the normalised sphere of the given radius, float32."""
    vs = _vox3(voxel_size)
    g = np.meshgrid(*[((sfft.fftfreq(n) * n) * v).astype(np.float32) for n, v in zip(shape, vs)],
                    indexing="ij", sparse=True)
    s = ((g[0] ** 2 + g[1] ** 2 + g[2] ** 2) <= np.float32(radius_mm ** 2)).astype(np.float32)
    s /= s.sum()
    return _rfft(s).real.astype(np.float32)


def _vsharp(field, mask, voxel_size, radii, threshold):
    shape = field.shape
    F = _rfft(field.astype(np.float32))
    M = _rfft(mask.astype(np.float32))
    out = np.zeros(shape, np.float32)
    done = np.zeros(shape, bool)
    inv = None
    delta = 1 - np.sqrt(1e-7)
    for r in sorted(radii, reverse=True):
        S = _smv_kr(shape, voxel_size, r)
        if inv is None:
            om = 1 - S
            small = np.abs(om) < threshold
            inv = np.where(small, 0, 1 / np.where(small, 1, om)).astype(np.float32)
            del om, small
        sel = _irfft(M * S, shape) > delta
        sel &= ~done
        hp = _irfft(F * (1 - S), shape)
        out[sel] = hp[sel]
        done |= sel
        del S, sel, hp
    del F, M
    loc = _irfft(_rfft(out) * inv, shape)
    loc[~done] = 0
    return loc, done


def _pdf(field, mask, voxel_size, b0_dir, tol, max_iter, pad):
    from scipy.sparse.linalg import LinearOperator, cg
    f = _pad(field.astype(np.float32), pad)
    m = _pad(mask, pad)
    shp = f.shape
    d = _dipole_r(shp, voxel_size, b0_dir)
    out = ~m
    n = int(out.sum())
    mf = m.astype(np.float32)

    def A(v):
        x = np.zeros(shp, np.float32)
        x[out] = v
        y = _irfft(_rfft(x) * d, shp)
        y *= mf
        return _irfft(_rfft(y) * d, shp)[out]

    b = _irfft(_rfft(f * mf) * d, shp)[out]
    op = LinearOperator((n, n), matvec=A, dtype=np.float32)
    try:
        xb, _ = cg(op, b, rtol=tol, maxiter=max_iter)
    except TypeError:            # scipy < 1.12
        xb, _ = cg(op, b, tol=tol, maxiter=max_iter)
    del b
    x = np.zeros(shp, np.float32)
    x[out] = xb
    del xb
    loc = (f - _irfft(_rfft(x) * d, shp)) * mf
    return _unpad(loc, pad), _unpad(m, pad)


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
    and set to 0); PDF keeps the whole mask but needs ``b0_dir``.  Background
    removal can only remove fields of sources outside the mask (harmonic inside
    it); judge it on data whose background satisfies that.
    """
    m = np.asarray(mask, bool)
    f = np.where(m, np.asarray(field, dtype=np.float32), np.float32(0))
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

def _diff_kr(shape):
    """Forward-difference multipliers on the half (rfft) spectrum."""
    E = []
    for ax, n in enumerate(shape):
        k = sfft.rfftfreq(n) if ax == 2 else sfft.fftfreq(n)
        s = [1, 1, 1]
        s[ax] = k.size
        E.append((np.exp(2j * np.pi * k) - 1).astype(np.complex64).reshape(s))
    return E


def _ete_r(shape):
    out = np.zeros(_rshape(shape), np.float32)
    for e in _diff_kr(shape):
        out += (np.abs(e) ** 2).astype(np.float32)
    return out


def _grad(x):
    return [np.roll(x, -1, axis=a) - x for a in range(3)]


def _grad_adj(g):
    out = np.roll(g[0], 1, axis=0) - g[0]
    for a in (1, 2):
        out += np.roll(g[a], 1, axis=a)
        out -= g[a]
    return out


def _tv_admm(phi, d, lam, rho, n_iter, tol):
    # single precision, half spectra: a whole-head volume stays well under 1 GB
    shape = phi.shape
    phi = phi.astype(np.float32)
    den = d * d + np.float32(rho) * _ete_r(shape)
    den[den == 0] = 1.0
    dphi = _rfft(phi) * d
    del phi
    x = np.zeros(shape, np.float32)
    zs = [np.zeros(shape, np.float32) for _ in range(3)]
    us = [np.zeros(shape, np.float32) for _ in range(3)]
    thr = np.float32(lam / rho)
    rho32 = np.float32(rho)
    for _ in range(n_iter):
        x_old = x
        q = _grad_adj([zs[a] - us[a] for a in range(3)])
        X = _rfft(q)
        del q
        X *= rho32
        X += dphi
        X /= den
        x = _irfft(X, shape)
        del X
        g = _grad(x)
        for a in range(3):
            v = g[a] + us[a]
            zs[a] = np.sign(v) * np.maximum(np.abs(v) - thr, np.float32(0))
            us[a] = v - zs[a]          # u + grad - z
            del v
        del g
        dx = np.linalg.norm(x - x_old) / max(np.linalg.norm(x), 1e-12)
        del x_old
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
    phi = _pad(np.where(m, np.asarray(local_field, dtype=np.float32), np.float32(0)), pad)
    shape = phi.shape
    d = _dipole_r(shape, vs, b0_dir)
    if method == "tv":
        lam = 3e-4 if lam is None else lam
        rho = 100 * lam if rho is None else rho
        chi = _tv_admm(phi, d, lam, rho, n_iter, tol)
    elif method == "tikhonov":
        lam = 1e-2 if lam is None else lam
        if regularization == "identity":
            g = np.float32(1.0)
        elif regularization == "gradient":
            g = _ete_r(shape)
        else:
            raise ValueError("regularization must be 'identity' or 'gradient'")
        den = d * d + np.float32(lam) * g
        den[den == 0] = 1.0
        chi = _irfft(_rfft(phi) * (d / den), shape)
    elif method == "tkd":
        big = np.abs(d) > tkd_threshold
        inv = np.where(big, 1 / np.where(big, d, 1), 0).astype(np.float32)
        chi = _irfft(_rfft(phi) * inv, shape)
    else:
        raise ValueError("method must be 'tv', 'tikhonov' or 'tkd'")
    del phi
    chi = np.ascontiguousarray(_unpad(chi, pad), dtype=np.float32)
    chi[~m] = 0
    return chi


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
                              voxel_size=vs, return_unwrapped=False)
    m1 = fe["reliable_mask"] if refine_mask else m0
    loc_hz, lm = qsm_background_removal(np.where(m1, fe["field_hz"], 0), m1, vs, method=background,
                                        b0_dir=b0_dir)
    scale = 1.0 / (gamma_bar_mhz_per_t * float(b0_tesla))   # Hz -> ppm
    loc_ppm = (loc_hz * np.float32(scale)).astype(np.float32)
    del loc_hz
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
    # brain, a thin CSF layer, dark skull (separates brain from scalp), scalp
    inner = ell((0, 0, 0), (0.72, 0.72, 0.72))
    skull = ell((0, 0, 0), (0.78, 0.78, 0.78)) & ~inner
    rho = np.where(inner, 0.9, np.where(head, 0.3, 0.0)).astype(np.float32)
    rho[brain] = 1.0
    rho[skull] = 0.03
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
