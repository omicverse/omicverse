"""Receptor-chain quality control for single-cell AIRR data.

``chain_qc`` classifies every cell by the configuration of its TCR / BCR
chains — a clean reimplementation of scirpy's ``tl.chain_qc``.  It is the
recommended first step after :func:`omicverse.airr.io.read_10x_vdj`: cells
with implausible chain configurations (multichains, orphan chains) can then
be filtered or flagged before clonotype definition.
"""
from __future__ import annotations

import pandas as pd

from .._registry import register_function


def _has(val) -> bool:
    """True if a chain slot is populated."""
    return val is not None and val == val and str(val) not in ("", "None", "nan")


def _classify_cell(row: pd.Series) -> tuple[str, str]:
    """Return ``(chain_pairing, receptor_subtype)`` for one cell."""
    vj1 = _has(row.get("VJ_1_junction_aa"))
    vj2 = _has(row.get("VJ_2_junction_aa"))
    vdj1 = _has(row.get("VDJ_1_junction_aa"))
    vdj2 = _has(row.get("VDJ_2_junction_aa"))

    n_vj = int(vj1) + int(vj2)
    n_vdj = int(vdj1) + int(vdj2)

    if n_vj == 0 and n_vdj == 0:
        pairing = "no IR"
    elif n_vj > 1 or n_vdj > 1:
        if (n_vj == 1 or n_vj == 0) and (n_vdj == 1 or n_vdj == 0):
            pairing = "single pair"
        elif n_vj <= 1 and n_vdj <= 1:
            pairing = "single pair"
        else:
            pairing = "multichain"
    elif n_vj == 1 and n_vdj == 1:
        pairing = "single pair"
    elif n_vj == 1 and n_vdj == 0:
        pairing = "orphan VJ"
    elif n_vj == 0 and n_vdj == 1:
        pairing = "orphan VDJ"
    else:  # n_vj > 1 or n_vdj > 1 already handled
        pairing = "extra chain"

    if n_vj > 1 or n_vdj > 1:
        pairing = "multichain"
    elif n_vj == 1 and n_vdj == 1:
        pairing = "single pair"
    elif (n_vj == 1) ^ (n_vdj == 1):
        pairing = "orphan VJ" if n_vj == 1 else "orphan VDJ"
    elif n_vj == 0 and n_vdj == 0:
        pairing = "no IR"

    # receptor subtype from loci
    loci = set()
    for slot in ("VJ_1", "VJ_2", "VDJ_1", "VDJ_2"):
        loc = row.get(f"{slot}_locus")
        if _has(loc):
            loci.add(str(loc).upper())
    if not loci:
        subtype = "no IR"
    elif loci <= {"TRA", "TRB"}:
        subtype = "TRA+TRB"
    elif loci <= {"TRG", "TRD"}:
        subtype = "TRG+TRD"
    elif loci <= {"IGH", "IGK"}:
        subtype = "IGH+IGK"
    elif loci <= {"IGH", "IGL"}:
        subtype = "IGH+IGL"
    elif loci <= {"IGH", "IGK", "IGL"}:
        subtype = "IGH+IGK/L"
    else:
        subtype = "ambiguous"
    return pairing, subtype


@register_function(
    aliases=["chain_qc", "airr_chain_qc", "免疫链质控", "链配对质控"],
    category="airr",
    description=(
        "Classify single-cell AIRR cells by their TCR/BCR chain "
        "configuration: 'single pair', 'orphan VJ', 'orphan VDJ', "
        "'multichain' or 'no IR'. Writes obs['chain_pairing'], "
        "obs['receptor_type'] and obs['receptor_subtype']."
    ),
    requires={"obs": ["VJ_1_junction_aa", "VDJ_1_junction_aa"]},
    produces={"obs": ["chain_pairing", "receptor_subtype"]},
    examples=[
        "ov.airr.chain_qc(adata)",
        "adata = adata[adata.obs['chain_pairing'] == 'single pair']",
    ],
    related=["airr.read_10x_vdj", "airr.define_clonotypes"],
)
def chain_qc(adata, *, inplace: bool = True):
    """Classify cells by their immune-receptor chain configuration.

    Parameters
    ----------
    adata
        AnnData produced by :func:`omicverse.airr.read_10x_vdj` /
        :func:`~omicverse.airr.read_airr`.
    inplace
        If ``True`` (default) annotate ``adata.obs`` in place; otherwise
        return the classification :class:`pandas.DataFrame`.

    Returns
    -------
    AnnData or :class:`pandas.DataFrame`
        ``adata.obs`` gains:

        ``chain_pairing``
            ``'single pair'`` / ``'orphan VJ'`` / ``'orphan VDJ'`` /
            ``'multichain'`` / ``'no IR'``.
        ``receptor_subtype``
            e.g. ``'TRA+TRB'``, ``'IGH+IGK'``.
        ``receptor_type``
            ``'TCR'`` / ``'BCR'`` / ``'ambiguous'`` / ``'no IR'``
            (kept from the reader if already present).
    """
    res = adata.obs.apply(_classify_cell, axis=1, result_type="expand")
    res.columns = ["chain_pairing", "receptor_subtype"]
    if not inplace:
        return res
    adata.obs["chain_pairing"] = res["chain_pairing"].values
    adata.obs["receptor_subtype"] = res["receptor_subtype"].values
    return adata
