r"""Transcripts meet segmentation: assignment, counting, transcript-based methods.

Imaging platforms (Xenium, CosMx, MERFISH/MERSCOPE, STARmap) and Stereo-seq
GEM files give one row per detected molecule. A label image turns them into
cells with :func:`assign_transcripts`; :func:`segment_transcripts` instead
segments from the molecules themselves (Baysor). :func:`aggregate` builds the
cell x gene AnnData either way.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from ..._optional import build_optional_dependency_error
from ..._registry import register_function


def _to_pixels(x, y, *, pixel_size=1.0, offset=(0.0, 0.0), transform=None):
    """Transcript coordinates -> fractional ``(col, row)`` pixel coordinates."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if transform is not None:
        A = np.asarray(transform, dtype=np.float64)
        if A.shape != (3, 3):
            raise ValueError("transform must be a 3x3 affine matrix (x, y, 1) -> (col, row, 1)")
        col = A[0, 0] * x + A[0, 1] * y + A[0, 2]
        row = A[1, 0] * x + A[1, 1] * y + A[1, 2]
        return col, row
    return (x - offset[0]) / pixel_size, (y - offset[1]) / pixel_size


def _from_pixels(col, row, *, pixel_size=1.0, offset=(0.0, 0.0), transform=None):
    """Inverse of :func:`_to_pixels`."""
    col = np.asarray(col, dtype=np.float64)
    row = np.asarray(row, dtype=np.float64)
    if transform is not None:
        inv = np.linalg.inv(np.asarray(transform, dtype=np.float64))
        return (inv[0, 0] * col + inv[0, 1] * row + inv[0, 2],
                inv[1, 0] * col + inv[1, 1] * row + inv[1, 2])
    return col * pixel_size + offset[0], row * pixel_size + offset[1]


@register_function(
    aliases=["转录本分配", "assign_transcripts", "transcript assignment", "分子分配到细胞"],
    category="space",
    description="Assign each detected transcript (Xenium/CosMx/MERFISH/Stereo-seq molecule table) to the segmented cell whose label covers it.",
    examples=[
        "tx['cell'] = ov.space.seg.assign_transcripts(tx, labels, pixel_size=0.2125)",
        "tx['cell'] = ov.space.seg.assign_transcripts(tx, labels, x='x_location', y='y_location', pixel_size=0.2125)",
    ],
    related=["space.seg.segment", "space.seg.aggregate"],
)
def assign_transcripts(
    transcripts: pd.DataFrame,
    labels,
    *,
    x: str = "x",
    y: str = "y",
    pixel_size: float = 1.0,
    offset: Sequence[float] = (0.0, 0.0),
    transform=None,
) -> np.ndarray:
    """Look up the label under every transcript.

    Parameters
    ----------
    transcripts
        One row per molecule with coordinate columns ``x`` and ``y``.
    labels
        Integer label image (dense or ``scipy.sparse``), e.g. from
        :func:`ov.space.seg.segment`.
    pixel_size, offset
        Transcript units per image pixel and the transcript coordinate of
        pixel ``(0, 0)``: ``col = (x - offset[0]) / pixel_size``. Xenium
        transcripts are in microns on a 0.2125 micron morphology grid.
    transform
        3x3 affine ``(x, y, 1) -> (col, row, 1)``; overrides ``pixel_size``
        and ``offset`` (e.g. a CosMx field-of-view or a registered H&E).

    Returns
    -------
    numpy.ndarray
        ``int64`` label per transcript; 0 = background or outside the image.
    """
    col, row = _to_pixels(transcripts[x].to_numpy(), transcripts[y].to_numpy(),
                          pixel_size=pixel_size, offset=offset, transform=transform)
    H, W = labels.shape[:2]
    c = np.floor(col).astype(np.int64)
    r = np.floor(row).astype(np.int64)
    inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    out = np.zeros(len(transcripts), dtype=np.int64)
    if hasattr(labels, "tocsr"):
        out[inside] = np.asarray(labels.tocsr()[r[inside], c[inside]]).ravel()
    else:
        out[inside] = np.asarray(labels)[r[inside], c[inside]]
    return out


def _unassigned(values: pd.Series) -> np.ndarray:
    """Background markers across tools: 0, -1, NaN, '', 'UNASSIGNED', 'noise'."""
    if pd.api.types.is_numeric_dtype(values):
        v = values.to_numpy()
        return ~np.isfinite(v.astype(float)) | (v <= 0)
    s = values.astype("string").fillna("")
    return s.str.strip().str.upper().isin(["", "0", "-1", "UNASSIGNED", "NOISE", "NAN"]).to_numpy()


def _polygons_wkt(labels, ids, *, pixel_size, offset, transform, tolerance=0.5) -> list[str]:
    """WKT outline of each label in ``ids`` in transcript coordinates."""
    from shapely.geometry import Polygon
    from skimage.measure import find_contours, regionprops

    dense = labels.toarray() if hasattr(labels, "toarray") else np.asarray(labels)
    wkt = {}
    for region in regionprops(dense):
        mask = np.pad(region.image, 1).astype(np.uint8)
        contours = find_contours(mask, 0.5)
        if not contours:
            continue
        ring = max(contours, key=len)
        if len(ring) < 4:
            continue
        r0, c0 = region.bbox[0] - 1, region.bbox[1] - 1
        # pixel (r, c) covers [c, c+1) x [r, r+1): centres sit at +0.5
        xs, ys = _from_pixels(ring[:, 1] + c0 + 0.5, ring[:, 0] + r0 + 0.5,
                              pixel_size=pixel_size, offset=offset, transform=transform)
        poly = Polygon(np.c_[xs, ys]).simplify(tolerance * pixel_size)
        if not poly.is_valid:
            poly = poly.buffer(0)
        wkt[region.label] = poly.wkt
    return [wkt.get(int(i), "") for i in ids]


@register_function(
    aliases=["转录本聚合", "aggregate", "cell by gene", "细胞表达矩阵", "transcripts_to_adata"],
    category="space",
    description="Count transcripts per segmented cell into a cell x gene AnnData with spatial centroids, cell area and WKT polygons ready for ov.pl.spatialseg.",
    examples=[
        "tx['cell'] = ov.space.seg.assign_transcripts(tx, labels, pixel_size=0.2125)",
        "adata = ov.space.seg.aggregate(tx, cell='cell', gene='feature_name', labels=labels, pixel_size=0.2125)",
        "adata = ov.space.seg.aggregate(baysor_tx, cell='cell', gene='gene')",
    ],
    related=["space.seg.assign_transcripts", "space.seg.segment_transcripts", "pl.spatialseg"],
)
def aggregate(
    transcripts: pd.DataFrame,
    *,
    cell: str = "cell",
    gene: str = "gene",
    x: str = "x",
    y: str = "y",
    count: str | None = None,
    labels=None,
    pixel_size: float = 1.0,
    offset: Sequence[float] = (0.0, 0.0),
    transform=None,
    geometry: bool | None = None,
    library_id: str = "seg",
):
    """Build a cell x gene count matrix from assigned transcripts.

    Parameters
    ----------
    transcripts
        Molecule table with a cell column (0 / -1 / NaN / '' = unassigned).
    cell, gene, x, y
        Column names.
    count
        Optional per-row count column (Stereo-seq ``MIDCount``); each row
        counts once otherwise.
    labels
        The label image the ``cell`` ids came from. Adds ``obs['area']``
        (transcript units squared) and, with ``geometry``,
        ``obs['geometry']`` WKT polygons for :func:`ov.pl.spatialseg`.
    pixel_size, offset, transform
        Same mapping as :func:`assign_transcripts`.
    geometry
        Write cell outlines (needs ``shapely``). ``None`` writes them when
        ``shapely`` is installed; ``True`` requires it.
    library_id
        Key under ``uns['spatial']``.

    Returns
    -------
    anndata.AnnData
        Raw counts in ``X`` (CSR), ``obs['n_transcripts']``,
        ``obs['n_genes']``, transcript centroids in ``obsm['spatial']``.
    """
    import anndata as ad
    from scipy import sparse

    tx = transcripts.loc[~_unassigned(transcripts[cell])]
    cell_codes, cell_ids = pd.factorize(tx[cell], sort=True)
    gene_codes, gene_ids = pd.factorize(tx[gene].astype(str), sort=True)
    w = tx[count].to_numpy(dtype=np.float32) if count else np.ones(len(tx), dtype=np.float32)
    X = sparse.csr_matrix((w, (cell_codes, gene_codes)),
                          shape=(len(cell_ids), len(gene_ids)), dtype=np.float32)
    X.sum_duplicates()

    n = np.bincount(cell_codes, minlength=len(cell_ids)).astype(np.float64)
    cx = np.bincount(cell_codes, weights=tx[x].to_numpy(dtype=np.float64), minlength=len(cell_ids)) / n
    cy = np.bincount(cell_codes, weights=tx[y].to_numpy(dtype=np.float64), minlength=len(cell_ids)) / n

    obs = pd.DataFrame(index=pd.Index(cell_ids.astype(str), name="cell_id"))
    obs["n_transcripts"] = np.asarray(X.sum(axis=1)).ravel()
    obs["n_genes"] = np.diff(X.indptr)
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(gene_ids, name="gene")))
    adata.obsm["spatial"] = np.c_[cx, cy]

    if labels is not None:
        ids = np.asarray(cell_ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("`labels` was given but the cell ids are not integer label ids")
        flat = labels.data if hasattr(labels, "tocsr") else np.asarray(labels).ravel()
        area_px = np.bincount(np.asarray(flat, dtype=np.int64), minlength=int(ids.max()) + 1)
        scale = abs(np.linalg.det(np.linalg.inv(np.asarray(transform, dtype=float))[:2, :2])) \
            if transform is not None else pixel_size ** 2
        adata.obs["area"] = area_px[ids] * scale
        if geometry is None:
            geometry = importlib.util.find_spec("shapely") is not None
        if geometry:
            try:
                import shapely  # noqa: F401
            except ImportError as exc:
                raise build_optional_dependency_error(
                    "ov.space.seg.aggregate(geometry=True)", ("shapely",),
                    install_hint="Install with `pip install shapely` or pass geometry=False.",
                ) from exc
            adata.obs["geometry"] = _polygons_wkt(labels, ids, pixel_size=pixel_size,
                                                  offset=offset, transform=transform)
    diam = 2 * np.sqrt(np.nanmedian(adata.obs["area"]) / np.pi) if "area" in adata.obs and adata.n_obs else 10.0
    adata.uns["spatial"] = {library_id: {
        "images": {},
        "scalefactors": {"tissue_hires_scalef": 1.0, "spot_diameter_fullres": float(diam)},
    }}
    return adata


# ---------------------------------------------------------------------------
# transcript-based segmentation
# ---------------------------------------------------------------------------
def _baysor_command(executable: str | Sequence[str]) -> list[str]:
    if isinstance(executable, (list, tuple)):
        return list(executable)
    found = shutil.which(executable)
    if found is None:
        raise FileNotFoundError(
            f"Baysor executable {executable!r} was not found on PATH. Install Baysor "
            "(https://kharchenkolab.github.io/Baysor/ — a prebuilt Linux binary or "
            "`julia -e 'using Pkg; Pkg.add(\"Baysor\")'`) or pass executable='/path/to/baysor'."
        )
    return [found]


def _run_baysor(tx, *, x, y, z, gene, prior, prior_confidence, scale, scale_std,
                min_molecules_per_cell, n_clusters, config, executable, extra_args,
                workdir, verbose):
    cols = [c for c in (x, y, z, gene, prior) if c is not None]
    tmp = None
    if workdir is None:
        tmp = tempfile.mkdtemp(prefix="ov_baysor_")
        workdir = tmp
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        inp = workdir / "molecules.csv"
        out = workdir / "segmentation.csv"
        frame = tx[cols].copy()
        if prior is not None:
            frame[prior] = np.where(_unassigned(frame[prior]), 0, pd.factorize(frame[prior])[0] + 1)
        frame.to_csv(inp, index=False)

        cmd = _baysor_command(executable) + ["run", "-x", x, "-y", y, "-g", gene,
                                             "-m", str(int(min_molecules_per_cell)),
                                             "-o", str(out)]
        if z is not None:
            cmd += ["-z", z]
        if scale is not None:
            cmd += ["-s", str(scale)]
        if scale_std is not None:
            cmd += ["--scale-std", str(scale_std)]
        if n_clusters is not None:
            cmd += ["--n-clusters", str(int(n_clusters))]
        if config is not None:
            cmd += ["-c", str(config)]
        if prior is not None:
            cmd += ["--prior-segmentation-confidence", str(prior_confidence)]
        cmd += list(extra_args or ())
        cmd += [str(inp)]
        if prior is not None:
            cmd += [f":{prior}"]

        proc = subprocess.run(cmd, cwd=str(workdir), capture_output=not verbose, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-2000:] if not verbose else ""
            raise RuntimeError(f"Baysor failed (exit {proc.returncode}): {' '.join(cmd)}\n{tail}")
        if not out.exists():
            raise RuntimeError(f"Baysor finished but wrote no {out}")
        return pd.read_csv(out)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


@register_function(
    aliases=["转录本分割", "segment_transcripts", "baysor", "基于转录本的细胞分割", "无图像分割"],
    category="space",
    description="Segment cells from the transcript table itself with Baysor (optionally using an image segmentation as a prior); returns the molecules with a cell column.",
    examples=[
        "tx = ov.space.seg.segment_transcripts(tx, method='baysor', gene='feature_name', scale=8)",
        "tx['nucleus'] = ov.space.seg.assign_transcripts(tx, nuclei, pixel_size=0.2125)",
        "tx = ov.space.seg.segment_transcripts(tx, method='baysor', prior='nucleus', prior_confidence=0.5)",
    ],
    related=["space.seg.aggregate", "space.seg.assign_transcripts", "space.seg.segment"],
)
def segment_transcripts(
    transcripts: pd.DataFrame,
    method: Literal["baysor"] = "baysor",
    *,
    x: str = "x",
    y: str = "y",
    z: str | None = None,
    gene: str = "gene",
    prior: str | None = None,
    prior_confidence: float = 0.2,
    scale: float | None = None,
    scale_std: str | float | None = "25%",
    min_molecules_per_cell: int = 30,
    n_clusters: int | None = None,
    config: str | None = None,
    executable: str | Sequence[str] = "baysor",
    extra_args: Sequence[str] = (),
    workdir: str | os.PathLike | None = None,
    key_added: str = "cell",
    verbose: bool = False,
) -> pd.DataFrame:
    """Segment cells from molecule positions and gene identities.

    Baysor (Petukhov et al., Nat. Biotechnol. 2022) fits a Markov random field
    mixture of cells over the molecules; it needs no image, and can refine an
    image segmentation given as a ``prior`` column. In the Nature Methods 2026
    benchmark it ranked among the best for coverage-adjusted performance.

    Parameters
    ----------
    transcripts
        One row per molecule.
    method
        ``'baysor'``.
    x, y, z, gene
        Column names (``z`` optional).
    prior
        Column with prior cell ids per molecule (0 / NaN = none), e.g. from
        :func:`assign_transcripts` on a nucleus segmentation.
    prior_confidence
        Baysor ``--prior-segmentation-confidence`` (0-1).
    scale, scale_std
        Expected cell radius in transcript units and its allowed spread.
        Baysor estimates ``scale`` when it is ``None`` and a prior is given.
    min_molecules_per_cell, n_clusters, config
        Baysor ``-m``, ``--n-clusters``, ``-c config.toml``.
    executable
        Baysor binary name/path, or a command list such as
        ``['julia', '--project=env', '-e', 'using Baysor; Baysor.command_main()', '--']``.
    extra_args
        Additional CLI arguments.
    workdir
        Keep Baysor's inputs and outputs here (a temporary directory otherwise).
    key_added
        Name of the output cell column.

    Returns
    -------
    pandas.DataFrame
        Copy of ``transcripts`` with ``key_added`` (int, 0 = unassigned or
        noise), ``f'{key_added}_confidence'`` and ``'is_noise'``.
    """
    if method != "baysor":
        raise ValueError(f"Unknown transcript method {method!r}; available: 'baysor'")
    if scale is None and prior is None:
        raise ValueError("Baysor needs `scale` (expected cell radius) unless a `prior` is given")
    seg = _run_baysor(transcripts, x=x, y=y, z=z, gene=gene, prior=prior,
                      prior_confidence=prior_confidence, scale=scale, scale_std=scale_std,
                      min_molecules_per_cell=min_molecules_per_cell, n_clusters=n_clusters,
                      config=config, executable=executable, extra_args=extra_args,
                      workdir=workdir, verbose=verbose)
    if len(seg) != len(transcripts):
        if "molecule_id" not in seg.columns:
            raise RuntimeError(f"Baysor returned {len(seg)} rows for {len(transcripts)} molecules")
        seg = seg.set_index("molecule_id").reindex(np.arange(1, len(transcripts) + 1)).reset_index()
    noise = (seg["is_noise"].astype(str).str.lower() == "true").to_numpy() \
        if "is_noise" in seg.columns else np.zeros(len(seg), dtype=bool)
    unassigned = _unassigned(seg["cell"]) | noise
    codes = pd.factorize(seg["cell"].where(~unassigned))[0].astype(np.int64) + 1
    codes[unassigned] = 0

    out = transcripts.copy()
    out[key_added] = codes
    if "assignment_confidence" in seg.columns:
        out[f"{key_added}_confidence"] = seg["assignment_confidence"].to_numpy()
    out["is_noise"] = noise
    return out
