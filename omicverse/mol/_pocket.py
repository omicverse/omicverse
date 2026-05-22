r"""Binding-pocket detection & druggability for :mod:`omicverse.mol`.

Backed by **rust-fpocket** (``fpocket-rs``) — a pip-installable Rust port
of fpocket's Voronoi / alpha-sphere pocket detection and its calibrated
druggability score. fpocket is the only native dependency in the
``ov.mol`` stack with no PyPI wheel, hence the Rust rewrite.

The ``fpocket_rs`` backend is expected to expose::

    fpocket_rs.detect_pockets(pdb_path: str) -> list[dict]

where each dict carries ``id, drug_score, volume, n_alpha_spheres,
hydrophobicity_score, polarity_score, residues, alpha_spheres``.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

import pandas as pd

from .._registry import register_function
from ._check import check_pocket

# fpocket's documented druggability cutoff — a pocket scoring above this is
# considered a viable small-molecule binding site
DRUGGABLE_CUTOFF = 0.5

_POCKET_COLUMNS = ["pocket_id", "rank", "drug_score", "volume",
                   "n_alpha_spheres", "hydrophobicity_score",
                   "polarity_score", "n_residues", "residues"]


def _get(d: Dict[str, Any], *names: str, default=None):
    """First present key among ``names`` (tolerates backend key spelling)."""
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


@register_function(
    aliases=["pockets", "结合口袋", "binding_pockets", "fpocket",
             "detect_pockets", "口袋检测", "cavities"],
    category="mol",
    description=(
        "Detect small-molecule binding pockets on a protein structure with "
        "the rust-fpocket backend (Voronoi alpha-spheres). Returns a ranked "
        "table — one row per pocket — with each pocket's druggability "
        "score, volume, alpha-sphere count, hydrophobicity, polarity and "
        "lining residues, and writes it onto structure.pockets so view() "
        "and dock() can reuse it."
    ),
    examples=[
        "df = ov.mol.pockets(s)",
        "df = ov.mol.pockets(s, min_drug_score=0.5)   # druggable only",
        "ov.mol.view(s, show_pockets=True)            # then overlay",
    ],
    related=["mol.druggability", "mol.view", "mol.dock"],
)
def pockets(structure, *, min_drug_score: float = 0.0,
            sort_by: str = "drug_score") -> pd.DataFrame:
    r"""Detect and rank binding pockets on a structure.

    Parameters
    ----------
    structure : MolStructure
        The protein to analyse.
    min_drug_score
        Drop pockets whose druggability score is below this.
    sort_by
        Column to rank pockets by (``'drug_score'`` or ``'volume'``).

    Returns
    -------
    pandas.DataFrame
        One row per pocket: ``pocket_id, rank, drug_score, volume,
        n_alpha_spheres, hydrophobicity_score, polarity_score, n_residues,
        residues``. Also written to ``structure.pockets``.
    """
    check_pocket()
    import fpocket_rs

    # Run fpocket on the cached structure file as-is. For an experimental
    # holo structure this keeps the bound ligand, which fpocket needs for
    # its solvent-accessibility / buriedness term — stripping it collapses
    # the druggability score of an otherwise-druggable pocket.
    if structure.path and os.path.exists(structure.path):
        raw = fpocket_rs.detect_pockets(structure.path)
    else:  # pragma: no cover - in-memory structure fallback
        fd, tmp = tempfile.mkstemp(suffix=".pdb")
        os.close(fd)
        try:
            structure.to_pdb(tmp)
            raw = fpocket_rs.detect_pockets(tmp)
        finally:
            os.unlink(tmp)

    # detect_pockets returns {'n_pockets': int, 'pockets': [...]}
    pocket_list = raw["pockets"] if isinstance(raw, dict) else list(raw)

    rows, spheres = [], {}
    for p in pocket_list:
        pid = int(_get(p, "pocket_id", "id", "pocket", default=0))
        residues = list(_get(p, "residues", "residue_ids", default=[]))
        rows.append({
            "pocket_id": pid,
            "rank": 0,
            "drug_score": float(_get(p, "drug_score", "druggability_score",
                                     default=0.0)),
            "volume": float(_get(p, "volume", default=0.0)),
            "n_alpha_spheres": int(_get(p, "n_alpha_spheres",
                                        "num_alpha_spheres", default=0)),
            "hydrophobicity_score": float(_get(p, "hydrophobicity_score",
                                               "hydrophobicity", default=0.0)),
            "polarity_score": float(_get(p, "polarity_score", "polarity",
                                         default=0.0)),
            "n_residues": len(residues),
            "residues": residues,
        })
        # alpha_spheres are (x, y, z, radius) — keep xyz for the box / view
        spheres[pid] = [(float(s[0]), float(s[1]), float(s[2]))
                        for s in _get(p, "alpha_spheres", "alpha_sphere_xyz",
                                      default=[])]

    df = pd.DataFrame(rows, columns=_POCKET_COLUMNS)
    if not df.empty:
        df = df[df["drug_score"] >= min_drug_score]
        col = sort_by if sort_by in df.columns else "drug_score"
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

    structure.pockets = df
    structure.meta["_pocket_spheres"] = {
        int(r.pocket_id): spheres.get(int(r.pocket_id), [])
        for r in df.itertuples()
    }
    return df


@register_function(
    aliases=["druggability", "成药性", "druggable", "druggability_score",
             "可成药性", "is_druggable"],
    category="mol",
    description=(
        "Summarise whether an omics target is druggable from its 3D "
        "structure — runs pocket detection, reports the top pocket's "
        "druggability score against fpocket's documented cutoff, and "
        "returns a druggable / difficult verdict with the ranked pocket "
        "table. The structure-based half of target prioritization."
    ),
    examples=[
        "verdict = ov.mol.druggability(s)",
        "verdict['druggable'], verdict['top_drug_score']",
    ],
    related=["mol.pockets", "mol.known_drugs"],
)
def druggability(structure) -> Dict[str, Any]:
    r"""Structure-based druggability verdict for a target.

    Parameters
    ----------
    structure : MolStructure
        The protein to assess.

    Returns
    -------
    dict
        ``top_drug_score`` — best pocket's druggability score;
        ``druggable`` — bool, ``top_drug_score >= 0.5``;
        ``verdict`` — ``'druggable'`` / ``'difficult'`` / ``'undruggable'``;
        ``n_pockets`` — number of detected pockets;
        ``pockets`` — the ranked pocket DataFrame.
    """
    df = structure.pockets
    if df is None:
        df = pockets(structure)

    if df.empty:
        return {"top_drug_score": 0.0, "druggable": False,
                "verdict": "undruggable", "n_pockets": 0, "pockets": df}

    top = float(df["drug_score"].max())
    if top >= DRUGGABLE_CUTOFF:
        verdict = "druggable"
    elif top >= 0.2:
        verdict = "difficult"
    else:
        verdict = "undruggable"
    return {
        "top_drug_score": top,
        "druggable": top >= DRUGGABLE_CUTOFF,
        "verdict": verdict,
        "n_pockets": int(len(df)),
        "pockets": df,
    }
