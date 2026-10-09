"""CellChatDB v2 tables, exported verbatim from R CellChat 2.2.0 (``CellChatDB.<species>``).

``data/<species>_interaction.csv.gz`` keeps the R rownames as ``id`` (the LR key
R uses as ``dimnames(net$prob)[[3]]``) plus the columns the inference and the
annotations need; ``complex`` and ``cofactor`` are complete. ``geneInfo`` is
only used by R to validate gene symbols and is not shipped.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

_DATA = Path(__file__).resolve().parent / "data"
SPECIES = ("human", "mouse", "zebrafish")
ANNOTATIONS = ("Secreted Signaling", "ECM-Receptor", "Non-protein Signaling", "Cell-Cell Contact")


@dataclass(frozen=True)
class CellChatDB:
    """CellChatDB: ``interaction`` (index ``id``), ``complex`` and ``cofactor`` tables."""

    species: str
    interaction: pd.DataFrame
    complex: pd.DataFrame
    cofactor: pd.DataFrame

    def subset(self, annotations: Sequence[str]) -> "CellChatDB":
        """R ``subsetDB(search = annotations, key = 'annotation')``."""
        unknown = set(annotations) - set(ANNOTATIONS)
        if unknown:
            raise ValueError(f"Unknown annotations {sorted(unknown)}; choose from {list(ANNOTATIONS)}.")
        keep = self.interaction["annotation"].isin(list(annotations))
        return replace(self, interaction=self.interaction[keep])

    def complex_subunits(self) -> Dict[str, List[str]]:
        cols = [c for c in self.complex.columns if c.startswith("subunit")]
        return {name: [g for g in row if g] for name, row in zip(self.complex.index, self.complex[cols].to_numpy())}

    def cofactor_genes(self, name: str) -> List[str]:
        if not name or name not in self.cofactor.index:
            return []
        cols = [c for c in self.cofactor.columns if c.startswith("cofactor")]
        return [g for g in self.cofactor.loc[name, cols] if g]


@lru_cache(maxsize=None)
def _load(species: str) -> CellChatDB:
    def read(table: str, **kw) -> pd.DataFrame:
        return pd.read_csv(_DATA / f"{species}_{table}.csv.gz", dtype=str, keep_default_na=False, **kw)

    interaction = read("interaction").set_index("id", drop=False)
    interaction.index.name = None
    return CellChatDB(species=species, interaction=interaction,
                      complex=read("complex", index_col="name"), cofactor=read("cofactor", index_col="name"))


def load_cellchat_db(species: str = "human") -> CellChatDB:
    """Load the bundled CellChatDB for ``'human'``, ``'mouse'`` or ``'zebrafish'``."""
    species = str(species).lower()
    if species not in SPECIES:
        raise ValueError(f"CellChatDB is available for {list(SPECIES)}; got {species!r}.")
    return _load(species)
