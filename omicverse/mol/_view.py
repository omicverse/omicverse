r"""Interactive 3D visualization for :mod:`omicverse.mol`.

Built on **py3Dmol** (3Dmol.js): each view is a self-contained HTML/JS
block embedded in the notebook cell output, with no ``ipywidgets``
dependency, and **renders in nbconvert HTML / mkdocs** without a live
kernel.

The one caveat is Jupyter's **notebook-trust model**. JupyterLab strips
``<script>`` (and ``<iframe>``) from outputs of notebooks it has not
signed with the user's local trust key — i.e. any notebook produced by
``nbconvert``, freshly cloned from a repo, or served by nbviewer. In
that case 3D views display only the "3Dmol.js failed to load" warning
that py3Dmol pre-inserts. The fix is to **trust the notebook** —
``jupyter trust <path>.ipynb`` — or simply re-execute the cells in your
own kernel (which auto-trusts outputs of the current session). The
mkdocs-rendered docs work without trust because static HTML is not
sanitized.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from .._registry import register_function
from ._check import _need, check_core

# AlphaFold's four official model-confidence bands (pLDDT)
_PLDDT_BANDS = [
    (90.0, 1e9, "#0053D6", "Very high (>90)"),
    (70.0, 90.0, "#65CBF3", "High (70-90)"),
    (50.0, 70.0, "#FFDB13", "Low (50-70)"),
    (-1e9, 50.0, "#FF7D45", "Very low (<50)"),
]


def _pdb_text(structure) -> str:
    """Serialize a MolStructure's atoms to a PDB string."""
    fd, tmp = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    try:
        structure.to_pdb(tmp)
        with open(tmp) as fh:
            return fh.read()
    finally:
        os.unlink(tmp)


def _value_colors(values: Mapping[int, float], cmap: str) -> Dict[int, str]:
    """Map ``{resid: value}`` to ``{resid: hex colour}`` via a colormap."""
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    vals = np.asarray(list(values.values()), dtype=float)
    finite = vals[np.isfinite(vals)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = mpl.colormaps[cmap]
    return {int(r): mcolors.to_hex(colormap(norm(v))) for r, v in values.items()}


def _apply_color(v, structure, color_by, cmap: str) -> Optional[str]:
    """Apply a ``color_by`` scheme to the cartoon; return a legend note."""
    resids = structure.residue_ids

    if isinstance(color_by, str) and color_by.lower() == "plddt":
        if structure.plddt is None:
            v.setStyle({}, {"cartoon": {"color": "spectrum"}})
            return "no pLDDT — coloured by spectrum"
        for lo, hi, hexcol, _ in _PLDDT_BANDS:
            sel = [int(r) for r, p in zip(resids, structure.plddt)
                   if lo <= p < hi]
            if sel:
                v.setStyle({"resi": sel}, {"cartoon": {"color": hexcol}})
        return "pLDDT confidence bands (blue high -> orange low)"

    if isinstance(color_by, str):
        scheme = {"chain": "chain", "spectrum": "spectrum",
                  "secondary_structure": "ssJmol", "ss": "ssJmol"}
        v.setStyle({}, {"cartoon": {"color": scheme.get(color_by.lower(),
                                                        color_by)}})
        return f"coloured by {color_by}"

    # a {resid: value} mapping or a per-residue array — the omics bridge
    if isinstance(color_by, Mapping):
        values = {int(k): float(v_) for k, v_ in color_by.items()}
    else:
        arr = np.asarray(list(color_by), dtype=float)
        if arr.shape[0] != len(resids):
            raise ValueError(
                f"color_by array has {arr.shape[0]} values but the structure "
                f"has {len(resids)} residues")
        values = {int(r): float(x) for r, x in zip(resids, arr)}
    colors = _value_colors(values, cmap)
    by_hex: Dict[str, list] = {}
    for resid, hexcol in colors.items():
        by_hex.setdefault(hexcol, []).append(resid)
    v.setStyle({}, {"cartoon": {"color": "lightgrey"}})
    for hexcol, sel in by_hex.items():
        v.setStyle({"resi": sel}, {"cartoon": {"color": hexcol}})
    return "coloured by per-residue value"


@register_function(
    aliases=["view", "结构可视化", "view_structure", "3d_view", "蛋白可视化",
             "show_structure", "interactive_structure"],
    category="mol",
    description=(
        "Interactive 3D visualization of a protein MolStructure with "
        "py3Dmol — a rotatable, zoomable view that renders inline in "
        "Jupyter and persists in nbconvert HTML exports. Colour by "
        "AlphaFold pLDDT confidence bands, by chain / secondary structure, "
        "or by any per-residue omics score (a {resid: value} dict or "
        "array); highlight residues (e.g. variant positions from "
        "ov.genetics) as sticks; overlay detected binding pockets."
    ),
    examples=[
        "ov.mol.view(s)                                  # pLDDT-coloured",
        "ov.mol.view(s, color_by='chain', surface=True)",
        "ov.mol.view(s, highlight=[790, 858], color_by='pLDDT')",
        "ov.mol.view(s, color_by=conservation_dict)      # omics score",
        "ov.mol.view(s, show_pockets=True)               # after pockets()",
    ],
    related=["mol.fetch_structure", "mol.view_docking", "mol.plot_pae"],
)
def view(structure, *, style: str = "cartoon", color_by: Any = "pLDDT",
         highlight: Optional[Sequence[int]] = None, show_pockets: bool = False,
         surface: bool = False, cmap: str = "viridis",
         width: int = 700, height: int = 500):
    r"""Render an interactive 3D view of a :class:`MolStructure`.

    Parameters
    ----------
    structure : MolStructure
        The structure to display.
    style
        Base representation — ``'cartoon'``, ``'stick'``, ``'sphere'`` or
        ``'line'``.
    color_by
        ``'pLDDT'`` (AlphaFold/ESMFold confidence bands), ``'chain'``,
        ``'secondary_structure'``, ``'spectrum'``, or a ``{resid: value}``
        mapping / per-residue array — the latter paints any omics-derived
        residue score.
    highlight
        Residue ids drawn as sticks on top of the cartoon (e.g. variant
        positions handed over from ``ov.genetics``).
    show_pockets
        Overlay detected binding pockets — requires ``ov.mol.pockets`` to
        have been run on ``structure`` first.
    surface
        Add a translucent molecular surface.
    cmap
        Matplotlib colormap used when ``color_by`` is a value mapping.
    width, height
        Viewport size in pixels.

    Returns
    -------
    py3Dmol.view
        Jupyter renders it automatically as the cell's last expression.
    """
    check_core()
    py3Dmol = _need("py3Dmol", "mol", "ov.mol interactive visualization")

    v = py3Dmol.view(width=width, height=height)
    v.addModel(_pdb_text(structure), "pdb")
    if style != "cartoon":
        v.setStyle({}, {style: {}})
        note = f"styled as {style}"
    else:
        note = _apply_color(v, structure, color_by, cmap)

    if highlight is not None:
        sel = {"resi": [int(r) for r in highlight]}
        v.addStyle(sel, {"stick": {"colorscheme": "orangeCarbon",
                                   "radius": 0.3}})
        v.addStyle(sel, {"sphere": {"colorscheme": "orangeCarbon",
                                    "radius": 0.4}})

    if show_pockets:
        spheres = (structure.meta or {}).get("_pocket_spheres", {})
        palette = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]
        for i, (pid, coords) in enumerate(spheres.items()):
            col = palette[i % len(palette)]
            for x, y, z in coords:
                v.addSphere({"center": {"x": float(x), "y": float(y),
                                        "z": float(z)},
                             "radius": 1.0, "color": col, "opacity": 0.55})

    if surface:
        v.addSurface(py3Dmol.VDW, {"opacity": 0.55, "color": "white"})

    v.zoomTo()
    if note:
        print(f"ov.mol.view: {note}")
    return v


@register_function(
    aliases=["view_docking", "对接可视化", "view_pose", "show_docking",
             "docking_view"],
    category="mol",
    description=(
        "Interactive 3D visualization of a docking pose — the protein "
        "cartoon plus a docked ligand pose as sticks, with the pose's Vina "
        "affinity annotated. Renders inline in Jupyter and persists in "
        "nbconvert HTML, like ov.mol.view."
    ),
    examples=[
        "result = ov.mol.dock(s, 'gefitinib', pocket=1)",
        "ov.mol.view_docking(s, result, pose=0)",
    ],
    related=["mol.dock", "mol.view"],
)
def view_docking(structure, result, *, pose: int = 0,
                 width: int = 700, height: int = 500):
    r"""Render a docked ligand pose inside its receptor.

    Parameters
    ----------
    structure : MolStructure
        The receptor.
    result : DockingResult
        Output of :func:`omicverse.mol.dock`.
    pose
        Pose index (0 = best-scoring).
    width, height
        Viewport size in pixels.

    Returns
    -------
    py3Dmol.view
    """
    check_core()
    py3Dmol = _need("py3Dmol", "mol", "ov.mol interactive visualization")
    if pose >= len(result.pose_blocks):
        raise IndexError(
            f"pose {pose} out of range — result has "
            f"{len(result.pose_blocks)} poses")

    v = py3Dmol.view(width=width, height=height)
    v.addModel(_pdb_text(structure), "pdb")
    v.setStyle({}, {"cartoon": {"color": "lightblue"}})

    v.addModel(result.pose_blocks[pose], "pdb")
    v.setStyle({"model": -1}, {"stick": {"colorscheme": "greenCarbon",
                                         "radius": 0.2}})

    if result.box is not None:
        center, size = result.box
        v.addBox({"center": {"x": float(center[0]), "y": float(center[1]),
                             "z": float(center[2])},
                  "dimensions": {"w": float(size[0]), "h": float(size[1]),
                                 "d": float(size[2])},
                  "color": "grey", "opacity": 0.18})

    aff = result.affinities[pose]
    v.addLabel(f"pose {pose}: {aff:.2f} kcal/mol",
               {"position": {"x": float(result.box[0][0]) if result.box
                             else 0.0,
                             "y": float(result.box[0][1]) if result.box
                             else 0.0,
                             "z": float(result.box[0][2]) if result.box
                             else 0.0},
                "backgroundColor": "white", "fontColor": "black",
                "fontSize": 12})
    v.zoomTo({"model": -1})
    return v


@register_function(
    aliases=["plot_pae", "pae", "predicted_aligned_error", "pae_plot",
             "误差矩阵"],
    category="mol",
    description=(
        "Plot the Predicted Aligned Error (PAE) matrix of an AlphaFold "
        "model as a heatmap. PAE is the model's estimate of inter-residue "
        "position error — high inter-domain PAE means the relative "
        "arrangement of two confidently-folded domains is itself "
        "uncertain, so structure-based drug work must stay within a "
        "single low-PAE domain."
    ),
    examples=[
        "ov.mol.plot_pae(s)",
    ],
    related=["mol.fetch_structure", "mol.view"],
)
def plot_pae(structure, *, ax=None, cmap: str = "Greens_r"):
    r"""Heatmap of the predicted-aligned-error matrix.

    Parameters
    ----------
    structure : MolStructure
        Must carry ``.pae`` (populated for AlphaFold DB models).
    ax : matplotlib.axes.Axes or None
        Draw into an existing axes instead of a new figure.
    cmap
        Matplotlib colormap (default reversed greens — dark = low error).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if structure.pae is None:
        raise ValueError(
            "structure has no PAE matrix — PAE is served for AlphaFold DB "
            "models; fetch with source='alphafold'.")
    pae = np.asarray(structure.pae, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(pae, cmap=cmap, origin="upper", vmin=0.0,
                   vmax=max(5.0, float(pae.max())))
    ax.set_xlabel("Aligned residue")
    ax.set_ylabel("Scored residue")
    ident = structure.gene or structure.uniprot or "structure"
    ax.set_title(f"Predicted aligned error — {ident}")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Expected position error (Angstrom)", rotation=90,
                       fontsize=8)
    return ax
