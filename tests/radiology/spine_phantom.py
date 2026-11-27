"""Synthetic sagittal spine phantom with known per-level alignment.

Vertebral bodies (rounded rectangles with a dark cortical rim and marrow
signal) and discs are placed along a lordotic curve that starts at the S1
endplate (sacral slope ``sacral_slope_deg``) and bends back by
``lordosis_deg`` over the lumbar column; the sacrum is an elongated tapered
block below. A translation ``slip_mm`` of every body above level ``slip_level``
(0 = the lowest disc, i.e. between the sacrum and the body above it) along
the lower body's superior endplate gives a known slip. Distractors: spinal
canal, posterior elements, subcutaneous fat, bowel-like blobs. Slice
thickness is simulated by averaging sub-slices across the lateral extent of
elliptical body cross-sections; acquisition effects are a multiplicative
bias field and Rician noise.

The ground truth per level is computed from the analytic (sharp) corners,
the Meyerding / Taillard definition: offset of the upper body's
posterior-inferior corner from the lower body's posterior-superior corner,
projected on the lower body's superior endplate, positive anterior.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

CONTRAST = {
    #            marrow cortex disc  disc_degen csf   soft  fat   bowel
    "T1": dict(marrow=0.70, cortex=0.12, disc=0.30, disc_degen=0.22, csf=0.15, soft=0.38, fat=0.95,
               bowel=(0.2, 0.6)),
    "T2": dict(marrow=0.45, cortex=0.10, disc=0.85, disc_degen=0.25, csf=1.00, soft=0.30, fat=0.80,
               bowel=(0.3, 0.9)),
    "CT": dict(marrow=0.80, cortex=1.00, disc=0.35, disc_degen=0.35, csf=0.30, soft=0.40, fat=0.25,
               bowel=(0.0, 0.45)),
}


def _rot(phi):
    """Unit column axis u (superior-ish) and endplate axis e (anterior) for a tilt
    ``phi`` (radians, positive = column leaning anteriorly going up); 2-D
    vectors in (anterior, superior) coordinates."""
    u = np.array([np.sin(phi), np.cos(phi)])
    e = np.array([np.cos(phi), -np.sin(phi)])
    return u, e


def _rounded_rect(c, u, e, h, d, r, n=6):
    pts = []
    r = min(r, h / 2 - 0.1, d / 2 - 0.1)
    for sx, sy, a0 in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        cx, cy = sx * (d / 2 - r), sy * (h / 2 - r)
        for a in np.linspace(a0, a0 + 90, n):
            t = np.deg2rad(a)
            x, y = cx + r * np.cos(t), cy + r * np.sin(t)
            pts.append(c + x * e + y * u)
    return np.array(pts)


def spine_geometry(*, n_lumbar: int = 5, body_h: float = 27.0, body_d: float = 38.0,
                   disc_h: float = 10.0, sacral_slope_deg: float = 38.0, lordosis_deg: float = 50.0,
                   slip_level: Optional[int] = 0, slip_mm: float = 0.0, retro: bool = False,
                   sacrum_len: float = 95.0) -> Dict:
    """Analytic geometry (mm, (anterior, superior) coordinates).

    Body ``k`` (k = 1..n_lumbar from the bottom) sits above disc ``k - 1``;
    body 0 is the sacrum's S1 segment.
    """
    total = n_lumbar * (body_h + disc_h)
    kappa = np.deg2rad(lordosis_deg) / total
    phi0 = np.deg2rad(sacral_slope_deg)
    u0, e0 = _rot(phi0)
    ps1 = np.array([0.0, 0.0])                      # S1 posterior-superior corner
    as1 = ps1 + body_d * e0
    sacrum = np.array([ps1, as1, as1 - sacrum_len * u0 - 0.35 * body_d * e0,
                       as1 - sacrum_len * u0 - 0.35 * body_d * e0 - 0.45 * body_d * e0,
                       ps1 - 0.75 * sacrum_len * u0])
    bodies = [{"label": "S1", "PS": ps1, "AS": as1, "u": u0, "e": e0, "kind": "sacrum"}]
    s_arc = 0.0
    top_c = ps1 + body_d / 2 * e0                    # superior endplate centre of the body below
    shift = np.zeros(2)
    for k in range(1, n_lumbar + 1):
        phi_disc = phi0 - kappa * (s_arc + disc_h / 2)
        ud, _ = _rot(phi_disc)
        inf_c = top_c + disc_h * ud
        phi_b = phi0 - kappa * (s_arc + disc_h + body_h / 2)
        u, e = _rot(phi_b)
        c = inf_c + body_h / 2 * u
        if slip_level is not None and k - 1 == slip_level and slip_mm:
            e_low = bodies[-1]["e"]
            shift = shift + (-1 if retro else 1) * slip_mm * e_low
        c = c + shift
        b = {"label": f"V{k}", "c": c, "u": u, "e": e, "kind": "body",
             "PI": c - body_h / 2 * u - body_d / 2 * e, "AI": c - body_h / 2 * u + body_d / 2 * e,
             "PS": c + body_h / 2 * u - body_d / 2 * e, "AS": c + body_h / 2 * u + body_d / 2 * e}
        bodies.append(b)
        top_c = c + body_h / 2 * u - shift          # the curve continues un-shifted
        s_arc += disc_h + body_h
    truth = []
    for k in range(1, len(bodies)):
        lo, up = bodies[k - 1], bodies[k]
        e_l = (lo["AS"] - lo["PS"]) / np.linalg.norm(lo["AS"] - lo["PS"])
        off = float((up["PI"] - lo["PS"]) @ e_l)
        L = float(np.linalg.norm(lo["AS"] - lo["PS"]))
        n_l = np.array([-e_l[1], e_l[0]])
        if n_l @ up["u"] < 0:
            n_l = -n_l
        dh = float(np.mean([(up["PI"] - lo["PS"]) @ n_l, (up["AI"] - lo["PS"]) @ n_l]))
        truth.append({"disc": k - 1, "offset_mm": off, "slip_pct": 100 * off / L, "lower_endplate_mm": L,
                      "disc_height_mm": dh})
    return {"bodies": bodies, "sacrum": sacrum, "truth": truth, "body_h": body_h, "body_d": body_d}


def _inset(poly: np.ndarray, t: float) -> np.ndarray:
    """Convex polygon shrunk by ``t`` mm (each edge moved inward, consecutive edges intersected)."""
    P = np.asarray(poly, float)
    n = len(P)
    ctr = P.mean(0)
    lines = []
    for k in range(n):
        a, b = P[k], P[(k + 1) % n]
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-d[1], d[0]])
        if (ctr - a) @ nrm < 0:
            nrm = -nrm
        lines.append((a + t * nrm, d))
    out = []
    for k in range(n):
        (p, r), (q, s_) = lines[k - 1], lines[k]
        A = np.array([r, -s_]).T
        out.append(p + np.linalg.solve(A, q - p)[0] * r)
    return np.array(out)


def _poly_mask(poly_px, shape, ss):
    from skimage.draw import polygon
    m = np.zeros((shape[0] * ss, shape[1] * ss), np.float32)
    rr, cc = polygon(poly_px[:, 0] * ss + (ss - 1) / 2, poly_px[:, 1] * ss + (ss - 1) / 2, m.shape)
    m[rr, cc] = 1
    return m.reshape(shape[0], ss, shape[1], ss).mean((1, 3))


def render_slice(geo: Dict, *, contrast: str = "T1", ps: float = 0.6, fov: float = 260.0,
                 x_mm: float = 0.0, thickness: float = 4.0, body_w: float = 46.0,
                 degenerate_disc: Optional[int] = None, ss: int = 2,
                 rng: Optional[np.random.Generator] = None, n_bowel: int = 6) -> np.ndarray:
    """Noise-free sagittal slice (rows = inferior, cols = posterior)."""
    rng = rng or np.random.default_rng(0)
    C = CONTRAST[contrast]
    n = int(round(fov / ps))
    shape = (n, n)
    # place the column: (anterior, superior) -> (row, col); centre of lumbar spine in the FOV
    allp = np.vstack([np.vstack([b["PS"], b["AS"]]) for b in geo["bodies"]] + [geo["sacrum"]])
    ctr = (allp.min(0) + allp.max(0)) / 2 + np.array([0.0, -5.0])
    origin_a, origin_s = ctr[0] + fov * 0.55, ctr[1] + fov / 2   # anterior of FOV left edge, top

    def px(P):
        P = np.atleast_2d(P)
        return np.stack([(origin_s - P[:, 1]) / ps, (origin_a - P[:, 0]) / ps], 1)

    sub = np.linspace(x_mm - thickness / 2, x_mm + thickness / 2, 5) if thickness > 0 else [x_mm]
    out = np.zeros(shape, np.float32)
    rr, cc = np.mgrid[0:n, 0:n]
    a_mm, s_mm = origin_a - cc * ps, origin_s - rr * ps
    bl = list(rng.uniform(0, 1, (n_bowel, 4)))
    for x in sub:
        f = np.sqrt(max(0.0, 1 - (2 * x / body_w) ** 2))
        img = np.zeros(shape, np.float32)
        # body outline: ellipse around the spine, air outside
        ba = (a_mm - (ctr[0] + 15)) / (fov * 0.47)
        bs = (s_mm - ctr[1]) / (fov * 0.75)
        inside = ba ** 2 + bs ** 2 < 1
        img[inside] = C["soft"]
        img[inside & (ba ** 2 + bs ** 2 > 0.80) & (ba < 0)] = C["fat"]     # posterior subcutaneous fat
        for (p, q, r, v) in bl:   # bowel-like blobs anteriorly
            ca, cs = ctr[0] + 60 + 50 * p, ctr[1] - 80 + 160 * q
            m = ((a_mm - ca) / (12 + 10 * r)) ** 2 + ((s_mm - cs) / (10 + 12 * v)) ** 2 < 1
            img[m & inside] = C["bowel"][0] + (C["bowel"][1] - C["bowel"][0]) * v
        bodies = geo["bodies"]
        # canal + posterior elements behind each lumbar body
        for b in bodies[1:]:
            c, u, e = b["c"], b["u"], b["e"]
            h, d = geo["body_h"], geo["body_d"]
            canal = _rounded_rect(c - (d / 2 + 8) * e, u, e, h + 14, 15, 1)
            img = np.maximum(img * (1 - _poly_mask(px(canal), shape, ss)), 0) + C["csf"] * _poly_mask(px(canal), shape, ss)
            spin = _rounded_rect(c - (d / 2 + 38) * e - 4 * u, u, e, 16, 30, 6)
            pm = _poly_mask(px(spin), shape, ss)
            img = img * (1 - pm) + C["cortex"] * pm
            pm2 = _poly_mask(px(_rounded_rect(c - (d / 2 + 38) * e - 4 * u, u, e, 12, 26, 5)), shape, ss)
            img = img * (1 - pm2) + C["marrow"] * pm2
        # discs
        for k in range(1, len(bodies)):
            lo, up = bodies[k - 1], bodies[k]
            mid_lo = (lo["PS"] + lo["AS"]) / 2
            mid_up = (up["PI"] + up["AI"]) / 2
            poly = np.array([mid_up + f * (up["PI"] - mid_up), mid_up + f * (up["AI"] - mid_up),
                             mid_lo + f * (lo["AS"] - mid_lo), mid_lo + f * (lo["PS"] - mid_lo)])
            m = _poly_mask(px(poly), shape, ss)
            v = C["disc_degen"] if degenerate_disc == k - 1 else C["disc"]
            img = img * (1 - m) + v * m
        # sacrum (cortex + marrow, faint S1-S2 remnant)
        sac = geo["sacrum"]
        sc = sac.mean(0)
        sac_f = sc + (sac - sc) * np.array([f, 1.0])
        m = _poly_mask(px(sac_f), shape, ss)
        img = img * (1 - m) + C["cortex"] * m
        inner = _inset(sac_f, 1.2)
        m = _poly_mask(px(inner), shape, ss)
        img = img * (1 - m) + C["marrow"] * m
        # vertebral bodies: elliptical cross-section -> AP depth scales with f
        for b in bodies[1:]:
            d = geo["body_d"] * f
            if d < 4:
                continue
            outer = _rounded_rect(b["c"], b["u"], b["e"], geo["body_h"], d, 3.0)
            m = _poly_mask(px(outer), shape, ss)
            img = img * (1 - m) + C["cortex"] * m
            innr = _rounded_rect(b["c"], b["u"], b["e"], geo["body_h"] - 2.4, d - 2.4, 2.0)
            m = _poly_mask(px(innr), shape, ss)
            img = img * (1 - m) + C["marrow"] * m
        out += img / len(sub)
    return out


def bias_field(shape, strength: float = 1.0, rng=None) -> np.ndarray:
    """exp(strength * smooth field): strength = log of the max/min ratio (about)."""
    rng = rng or np.random.default_rng(1)
    n0, n1 = shape
    r, c = np.mgrid[0:n0, 0:n1]
    th = rng.uniform(0, 2 * np.pi)
    lin = ((r / n0 - 0.5) * np.cos(th) + (c / n1 - 0.5) * np.sin(th))
    br, bc = rng.uniform(0.2, 0.8, 2)
    bump = np.exp(-(((r / n0 - br) ** 2 + (c / n1 - bc) ** 2) / 0.08))
    f = lin + 0.5 * bump
    f = (f - f.min()) / (np.ptp(f) + 1e-9) - 0.5
    return np.exp(strength * f)


def acquire(img: np.ndarray, *, noise: float = 0.03, bias: float = 0.0, rng=None, scale: float = 1000.0):
    rng = rng or np.random.default_rng(2)
    x = img * (bias_field(img.shape, bias, rng) if bias else 1.0)
    if noise:
        s = noise * float(np.percentile(img[img > 0], 90))
        x = np.sqrt((x + rng.normal(0, s, x.shape)) ** 2 + rng.normal(0, s, x.shape) ** 2)
    return np.clip(x * scale, 0, 65000)


def write_sag_series(root, desc, slices: List[np.ndarray], *, ps: float, spacing: float, x0: float = 0.0,
                     iop=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0), origin=(0.0, -130.0, 130.0), number: int = 1,
                     thickness: Optional[float] = None, modality: str = "MR"):
    """Write slices (lateral positions x0 + k * spacing, k centred) as a DICOM series."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    iop = np.asarray(iop, float)
    normal = np.cross(iop[:3], iop[3:])
    suid = generate_uid()
    d = root / desc.replace(" ", "_")
    d.mkdir(parents=True)
    n = len(slices)
    for k, px in enumerate(slices):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        sop = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = sop
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        ds.SOPInstanceUID = sop
        ds.SeriesInstanceUID = suid
        ds.StudyInstanceUID = "1.2.3.4"
        ds.SeriesDescription = desc
        ds.SeriesNumber = number
        ds.InstanceNumber = k + 1
        ds.Modality = modality
        ds.ImageOrientationPatient = [float(v) for v in iop]
        pos = np.asarray(origin, float) + (x0 + (k - (n - 1) / 2) * spacing) * normal
        ds.ImagePositionPatient = [float(v) for v in pos]
        ds.PixelSpacing = [ps, ps]
        if thickness:
            ds.SliceThickness = float(thickness)
        ds.Rows, ds.Columns = px.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelData = np.asarray(px).astype(np.uint16).tobytes()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(str(d / f"IM{k:03d}"), write_like_original=False)
    return suid


def make_study(root, *, contrast: str = "T1", slip_level: Optional[int] = 0, slip_mm: float = 0.0,
               retro: bool = False, noise: float = 0.03, bias: float = 0.0, thickness: float = 4.0,
               lordosis_deg: float = 50.0, sacral_slope_deg: float = 38.0, n_slices: int = 5,
               spacing: float = 4.0, ps: float = 0.6, x_centre: float = 0.0, seed: int = 0,
               desc: Optional[str] = None, degenerate_disc: Optional[int] = None, n_bowel: int = 6,
               iop=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0)):
    """Write a phantom sagittal series; returns (series_dir_root, geometry)."""
    geo = spine_geometry(slip_level=slip_level, slip_mm=slip_mm, retro=retro, lordosis_deg=lordosis_deg,
                         sacral_slope_deg=sacral_slope_deg)
    rng = np.random.default_rng(seed)
    sl = []
    for k in range(n_slices):
        x = x_centre + (k - (n_slices - 1) / 2) * spacing
        im = render_slice(geo, contrast=contrast, ps=ps, x_mm=x, thickness=thickness, rng=np.random.default_rng(seed),
                          degenerate_disc=degenerate_disc, n_bowel=n_bowel)
        sl.append(acquire(im, noise=noise, bias=bias, rng=rng))
    write_sag_series(root, desc or f"Sag {contrast}", sl, ps=ps, spacing=spacing, thickness=thickness, iop=iop,
                     modality="CT" if contrast == "CT" else "MR")
    return geo
