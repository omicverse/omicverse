r"""Reference-free segmentation quality metrics.

The subset of the CellSPA metrics used by Martinez-de-Morentin et al.
(Nat. Methods 2026) that needs no single-cell reference: segmentation yield
and transcript recovery (overall characteristics), transcripts and genes per
cell (cell-level QC), and cell morphology (area, elongation, circularity,
sphericity, compactness, convexity, eccentricity, solidity).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ..._registry import register_function
from ._transcripts import _unassigned

_MORPHOLOGY = ["area", "elongation", "circularity", "sphericity", "compactness",
               "convexity", "eccentricity", "solidity"]


def _morphology(labels) -> pd.DataFrame:
    """Per-cell shape descriptors; formulas follow the benchmark's Methods (M5-M12)."""
    from scipy import ndimage as ndi
    from skimage.measure import perimeter, regionprops

    dense = labels.toarray() if hasattr(labels, "toarray") else np.asarray(labels)
    rows = []
    for r in regionprops(dense):
        area = float(r.area)
        p = float(r.perimeter) or np.sqrt(area) * 4
        p_convex = float(perimeter(r.image_convex)) or p
        h = r.bbox[2] - r.bbox[0]
        w = r.bbox[3] - r.bbox[1]
        major = float(r.axis_major_length)
        minor = float(r.axis_minor_length)
        # inscribed radius: deepest pixel to the nearest background pixel;
        # circumscribed radius: half the maximum Feret diameter
        r_in = float(ndi.distance_transform_edt(np.pad(r.image, 1)).max())
        r_circ = max(float(r.feret_diameter_max) / 2.0, 0.5)
        rows.append({
            "label": r.label,
            "area": area,
            # bounding-box aspect ratio, >= 1 so it is rotation-symmetric in x/y
            "elongation": max(h, w) / max(min(h, w), 1),
            "circularity": min(4 * np.pi * area / p_convex ** 2, 1.0),
            "sphericity": min(r_in / r_circ, 1.0),
            "compactness": min(4 * np.pi * area / p ** 2, 1.0),
            "convexity": min(p_convex / p, 1.0),
            "eccentricity": major / minor if minor > 0 else np.nan,
            "solidity": float(r.solidity),
        })
    return pd.DataFrame(rows, columns=["label"] + _MORPHOLOGY).set_index("label")


@register_function(
    aliases=["分割评估", "segmentation metrics", "evaluate segmentation", "CellSPA", "细胞形态指标"],
    category="space",
    description="Per-cell segmentation quality metrics (CellSPA subset from the Nature Methods 2026 benchmark): transcripts and genes per cell, area, elongation, circularity, sphericity, compactness, convexity, eccentricity, solidity.",
    examples=[
        "cells = ov.space.seg.evaluate(labels, transcripts=tx, cell='cell', gene='gene')",
        "cells = ov.space.seg.evaluate(labels)  # morphology only",
    ],
    related=["space.seg.compare", "space.seg.segment"],
)
def evaluate(
    labels=None,
    transcripts: pd.DataFrame | None = None,
    *,
    cell: str = "cell",
    gene: str = "gene",
    count: str | None = None,
    pixel_size: float = 1.0,
) -> pd.DataFrame:
    """Per-cell quality metrics of one segmentation.

    Parameters
    ----------
    labels
        Label image; enables the morphology columns. ``None`` for a
        transcript-only segmentation (Baysor).
    transcripts
        Molecules with a ``cell`` column (0 / NaN = unassigned); enables
        ``n_transcripts`` and ``n_genes``. The overall assignment rate is
        stored in ``.attrs['pct_assigned']``.
    count
        Optional per-row count column.
    pixel_size
        Converts ``area`` from pixels to transcript units squared.

    Returns
    -------
    pandas.DataFrame
        One row per cell, indexed by cell id.
    """
    if labels is None and transcripts is None:
        raise ValueError("give `labels`, `transcripts`, or both")
    parts = []
    if labels is not None:
        morph = _morphology(labels)
        morph["area"] *= pixel_size ** 2
        morph.index = morph.index.astype(str)
        parts.append(morph)
    pct = None
    if transcripts is not None:
        miss = _unassigned(transcripts[cell])
        w = transcripts[count].to_numpy(dtype=float) if count else np.ones(len(transcripts))
        pct = 100.0 * w[~miss].sum() / max(w.sum(), 1e-12)
        tx = transcripts.loc[~miss]
        ids = tx[cell].astype(str) if labels is None else tx[cell].astype(np.int64).astype(str)
        grouped = pd.DataFrame({"cell": ids.to_numpy(), "gene": tx[gene].to_numpy(),
                                "w": w[~miss]}).groupby("cell", sort=False)
        qc = pd.DataFrame({"n_transcripts": grouped["w"].sum(),
                           "n_genes": grouped["gene"].nunique()})
        parts.append(qc)
    out = pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]
    if transcripts is not None:
        out[["n_transcripts", "n_genes"]] = out[["n_transcripts", "n_genes"]].fillna(0)
    out.index.name = "cell"
    out.attrs["pct_assigned"] = pct
    return out


def _summary(cells: pd.DataFrame, *, min_transcripts, max_transcripts, min_genes) -> dict:
    row = {"n_cells": len(cells)}
    if "n_transcripts" in cells:
        hq = ((cells["n_transcripts"] >= min_transcripts)
              & (cells["n_transcripts"] <= max_transcripts)
              & (cells["n_genes"] > min_genes))
        row["n_hq_cells"] = int(hq.sum())
        row["pct_assigned"] = cells.attrs.get("pct_assigned")
        row["transcripts_per_cell"] = float(cells.loc[hq, "n_transcripts"].mean()) if hq.any() else np.nan
        row["genes_per_cell"] = float(cells.loc[hq, "n_genes"].mean()) if hq.any() else np.nan
    for m in _MORPHOLOGY:
        if m in cells:
            row[m] = float(cells[m].median())
    return row


@register_function(
    aliases=["分割方法比较", "compare segmentation", "segmentation benchmark", "分割基准"],
    category="space",
    description="Compare several segmentations of the same sample on reference-free CellSPA metrics (high-quality cell count, % transcripts assigned, transcripts/genes per cell, morphology medians) with a min-max-scaled overall rank, as in the Nature Methods 2026 benchmark.",
    examples=[
        "res = {m: ov.space.seg.evaluate(lab, transcripts=tx.assign(cell=ov.space.seg.assign_transcripts(tx, lab))) for m, lab in labels.items()}",
        "table = ov.space.seg.compare(res)",
    ],
    related=["space.seg.evaluate"],
)
def compare(
    results: Mapping[str, pd.DataFrame],
    *,
    min_transcripts: int = 20,
    max_transcripts: int = 2000,
    min_genes: int = 20,
    score: tuple[str, ...] = ("n_hq_cells", "pct_assigned", "transcripts_per_cell",
                              "genes_per_cell"),
) -> pd.DataFrame:
    """Side-by-side summary of segmentations, ranked.

    Parameters
    ----------
    results
        ``{method_name: ov.space.seg.evaluate(...)}``.
    min_transcripts, max_transcripts, min_genes
        High-quality cell filter (the benchmark's: 20-2,000 transcripts and
        more than 20 genes).
    score
        Columns that enter the overall score; each is min-max scaled across
        methods (higher is better) and averaged, as in the benchmark.

    Returns
    -------
    pandas.DataFrame
        One row per method with the summary metrics, ``overall`` and ``rank``.
    """
    table = pd.DataFrame({name: _summary(cells, min_transcripts=min_transcripts,
                                         max_transcripts=max_transcripts, min_genes=min_genes)
                          for name, cells in results.items()}).T
    table = table.apply(pd.to_numeric, errors="coerce")
    used = [c for c in score if c in table and table[c].notna().any()]
    if used:
        sub = table[used]
        span = (sub.max() - sub.min()).replace(0, np.nan)
        scaled = ((sub - sub.min()) / span).fillna(1.0)
        table["overall"] = scaled.mean(axis=1)
        table["rank"] = table["overall"].rank(ascending=False, method="min").astype(int)
        table = table.sort_values("rank")
    table.index.name = "method"
    return table
