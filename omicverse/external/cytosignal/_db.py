"""CytoSignal ligand-receptor database (exported from the R package data)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_DATA = Path(__file__).with_name("data")
TYPES = ("diffusion", "contact")


@dataclass(frozen=True)
class CytoSignalDB:
    """``g_to_u`` / ``db.diff`` / ``db.cont`` / ``inter.index`` of R CytoSignal.

    ``interactions`` is the long form of the R factors: one row per
    (``db``, ``slot``, ``uniprot``, ``interaction``) in the original order,
    where ``db`` is ``'diffusion'`` (``db.diff``) or ``'contact'``
    (``db.cont``) and ``slot`` one of ``combined`` / ``ligands`` / ``receptors``.
    """

    gene_to_uniprot: pd.DataFrame
    interactions: pd.DataFrame
    inter_index: pd.DataFrame

    def components(self, db: str) -> dict[str, dict[str, list[str]]]:
        """``{interaction: {'combined'|'ligands'|'receptors': [uniprot, ...]}}``
        in factor-level order (the order R uses for score columns)."""
        tab = self.interactions[self.interactions["db"] == db]
        out: dict[str, dict[str, list[str]]] = {}
        for slot in ("combined", "ligands", "receptors"):
            sub = tab[tab["slot"] == slot]
            for intr, uni in zip(sub["interaction"], sub["uniprot"]):
                out.setdefault(intr, {"combined": [], "ligands": [], "receptors": []})[slot].append(uni)
        return out

    @property
    def uniprot_to_gene(self) -> dict[str, str]:
        return dict(zip(self.gene_to_uniprot["uniprot"], self.gene_to_uniprot["hgnc_symbol"]))


@lru_cache(maxsize=1)
def load_cytosignal_db() -> CytoSignalDB:
    """Load the CytoSignal (CellPhoneDB-derived, human gene symbols) database."""
    g2u = pd.read_csv(_DATA / "cytosignal_gene_to_uniprot.csv.gz", dtype=str, keep_default_na=False)
    tab = pd.read_csv(_DATA / "cytosignal_interactions.csv.gz", dtype=str, keep_default_na=False)
    idx = pd.read_csv(_DATA / "cytosignal_inter_index.csv.gz", keep_default_na=False)
    idx.index = idx["id_cp_interaction"].astype(str).to_numpy()
    return CytoSignalDB(gene_to_uniprot=g2u, interactions=tab, inter_index=idx)


def to_uniprot(var_names, db: CytoSignalDB):
    """R ``changeUniprot``: upper-case gene symbols, keep database genes.

    Returns ``(gene_index, uniprot_ids)``: column positions in ``var_names``
    (data order) and their UniProt IDs. Duplicated upper-cased symbols keep
    their first occurrence, as R's ``object[uniprot.genes, ]`` does.
    """
    g2u = dict(zip(db.gene_to_uniprot["gene_name"], db.gene_to_uniprot["uniprot"]))
    upper = pd.Index([str(v).upper() for v in var_names])
    first = {}
    for i, g in enumerate(upper):
        first.setdefault(g, i)
    keep = [i for i, g in enumerate(upper) if g in g2u]
    # R selects rows by name, so a duplicated symbol maps to its first row twice;
    # the duplicated UniProt rows are then dropped (keeping the first).
    rows, seen = [], set()
    for i in keep:
        u = g2u[upper[i]]
        if u in seen:
            continue
        seen.add(u)
        rows.append(first[upper[i]])
    return np.asarray(rows, dtype=np.int64), [g2u[upper[i]] for i in rows]
