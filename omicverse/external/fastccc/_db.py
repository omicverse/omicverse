"""CellPhoneDB-format ligand-receptor database used by FastCCC."""
from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass

import pandas as pd

_TABLES = ("interaction_table", "gene_table", "protein_table", "complex_table",
           "complex_composition_table", "multidata_table")


@dataclass(frozen=True)
class CPDBTables:
    """The CellPhoneDB v4/v5 csv tables FastCCC reads (``cellphonedb.zip`` or its folder)."""

    interaction_table: pd.DataFrame
    gene_table: pd.DataFrame
    protein_table: pd.DataFrame
    complex_table: pd.DataFrame
    complex_composition_table: pd.DataFrame
    multidata_table: pd.DataFrame
    source: str = ""

    # ------------------------------------------------------------------ FastCCC inputs
    def interactions(self) -> pd.DataFrame:
        """``id_cp_interaction`` -> (multidata_1_id, multidata_2_id), duplicates dropped."""
        it = self.interaction_table[["id_cp_interaction", "multidata_1_id", "multidata_2_id"]]
        return it.set_index("id_cp_interaction")[["multidata_1_id", "multidata_2_id"]].drop_duplicates()

    def gene_to_protein(self, convert_type: str = "hgnc_symbol") -> pd.DataFrame:
        """Gene identifier -> protein multidata id, in database order."""
        merged = self.gene_table.merge(self.protein_table, left_on="protein_id", right_on="id_protein")
        tmp = merged[[convert_type, "protein_multidata_id"]].drop_duplicates()
        return tmp.set_index("protein_multidata_id")

    def complex_members(self) -> pd.DataFrame:
        """(complex_multidata_id, protein_multidata_id) rows."""
        ct = self.complex_table.merge(self.complex_composition_table, on="complex_multidata_id")
        return ct[["complex_multidata_id", "protein_multidata_id"]]

    # ------------------------------------------------------------------ annotation
    def partner_names_and_genes(self) -> tuple[dict, dict]:
        """multidata id -> display name and -> sorted subunit gene symbols.

        Single proteins are named by their gene symbol (CellPhoneDB's
        ``interacting_pair`` convention), complexes by the complex name.
        """
        # gene_table.protein_id is the protein's id_multidata (protein_table.id_protein == protein_multidata_id)
        merged = self.gene_table.merge(self.protein_table, left_on="protein_id", right_on="id_protein")
        prot_gene = merged.drop_duplicates("protein_multidata_id").set_index("protein_multidata_id")["gene_name"]
        md = self.multidata_table.set_index("id_multidata")
        members = self.complex_members().groupby("complex_multidata_id")["protein_multidata_id"].apply(list)
        names, genes = {}, {}
        for mid, row in md.iterrows():
            if bool(row["is_complex"]):
                names[mid] = str(row["name"])
                genes[mid] = tuple(sorted({str(prot_gene[p]) for p in members.get(mid, []) if p in prot_gene.index}))
            else:
                g = str(prot_gene[mid]) if mid in prot_gene.index else str(row["name"])
                names[mid], genes[mid] = g, (g,)
        return names, genes


def load_cpdb_tables(path: str) -> CPDBTables:
    """Read the CellPhoneDB tables from ``cellphonedb.zip`` or an extracted folder."""
    path = os.fspath(path)
    frames = {}
    if os.path.isdir(path):
        for name in _TABLES:
            frames[name] = pd.read_csv(os.path.join(path, f"{name}.csv"))
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            members = {os.path.basename(n): n for n in zf.namelist()}
            for name in _TABLES:
                if f"{name}.csv" not in members:
                    raise ValueError(f"{path} is not a CellPhoneDB database: missing {name}.csv")
                with zf.open(members[f"{name}.csv"]) as fh:
                    frames[name] = pd.read_csv(fh)
    else:
        raise FileNotFoundError(f"CellPhoneDB database not found (zip or folder): {path}")
    return CPDBTables(**frames, source=path)
