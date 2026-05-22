r"""Protein-structure acquisition for :mod:`omicverse.mol`.

:class:`MolStructure` is the central object — a parsed protein structure
plus its confidence metadata. :func:`fetch_structure` pulls an
experimental (RCSB PDB) or predicted (AlphaFold DB) structure;
:func:`predict_structure` folds a raw sequence with the ESMFold API.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import numpy as np

from .._registry import register_function
from ._check import _need, check_core

# canonical AlphaFold DB / RCSB / ESMFold endpoints. The AlphaFold DB
# file version is bumped periodically (v4 -> v6 ...), so model + PAE URLs
# are resolved from the prediction API rather than hard-coded.
_AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
_RCSB = "https://files.rcsb.org/download/{pdb}.pdb"
_UNIPROT = ("https://rest.uniprot.org/uniprotkb/search"
            "?query=gene_exact:{gene}+AND+organism_id:{taxon}+AND+reviewed:true"
            "&fields=accession&format=json&size=1")
_ESMFOLD = "https://api.esmatlas.com/foldSequence/v1/pdb/"

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# UniProt accession pattern (the official regex, simplified)
_UNIPROT_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)
_PDB_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


class MolStructure:
    """A parsed protein structure with confidence metadata.

    Attributes
    ----------
    atoms : biotite.structure.AtomArray
        Coordinates, elements, residue / chain identifiers.
    gene, uniprot : str or None
        Identifiers, when resolved.
    sequence : str
        One-letter amino-acid sequence.
    source : {'pdb', 'alphafold', 'esmfold'}
        Where the structure came from.
    plddt : numpy.ndarray or None
        Per-residue model confidence (AlphaFold / ESMFold), 0-100.
    pae : numpy.ndarray or None
        Predicted-aligned-error matrix (AlphaFold DB), residues x residues.
    pockets : pandas.DataFrame or None
        Written by :func:`omicverse.mol.pockets`.
    path : str
        Local cached structure file.
    meta : dict
        Method, resolution, organism, model metadata.
    """

    def __init__(self, atoms, *, source, path, gene=None, uniprot=None,
                 plddt=None, pae=None, meta=None):
        self.atoms = atoms
        self.source = source
        self.path = path
        self.gene = gene
        self.uniprot = uniprot
        self.plddt = plddt
        self.pae = pae
        self.pockets = None
        self.meta = meta or {}
        self._ca = atoms[atoms.atom_name == "CA"]

    @property
    def sequence(self) -> str:
        """One-letter amino-acid sequence (from CA atoms)."""
        return "".join(_THREE_TO_ONE.get(r, "X") for r in self._ca.res_name)

    @property
    def n_residues(self) -> int:
        return len(self._ca)

    @property
    def residue_ids(self) -> np.ndarray:
        """Residue numbers, in structure order."""
        return np.asarray(self._ca.res_id)

    def to_pdb(self, path: str) -> str:
        """Write the structure to a PDB file; return the path."""
        pdb = _need("biotite.structure.io.pdb", "mol", "ov.mol")
        f = pdb.PDBFile()
        f.set_structure(self.atoms)
        f.write(path)
        return path

    def copy(self) -> "MolStructure":
        s = MolStructure(
            self.atoms.copy(), source=self.source, path=self.path,
            gene=self.gene, uniprot=self.uniprot,
            plddt=None if self.plddt is None else self.plddt.copy(),
            pae=None if self.pae is None else self.pae.copy(),
            meta=dict(self.meta),
        )
        s.pockets = None if self.pockets is None else self.pockets.copy()
        return s

    def __repr__(self) -> str:
        ident = self.gene or self.uniprot or self.meta.get("pdb_id") or "?"
        conf = ""
        if self.plddt is not None:
            conf = f", mean pLDDT {float(np.mean(self.plddt)):.1f}"
        return (f"MolStructure({ident}, source={self.source}, "
                f"{self.n_residues} residues{conf})")


# ------------------------------------------------------------------ #
# helpers                                                             #
# ------------------------------------------------------------------ #


def _download(url: str, path: str, *, timeout: int = 60) -> str:
    """Download ``url`` to ``path`` (skip if already cached)."""
    import requests
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(path, "wb") as fh:
        fh.write(r.content)
    return path


def _gene_to_uniprot(gene: str, taxon: int = 9606) -> str:
    """Resolve a gene symbol to its reviewed UniProt accession."""
    import requests
    r = requests.get(_UNIPROT.format(gene=gene, taxon=taxon), timeout=60)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise ValueError(
            f"could not resolve gene {gene!r} to a reviewed UniProt accession "
            f"(taxon {taxon}). Pass a UniProt accession or PDB ID directly."
        )
    return results[0]["primaryAccession"]


def _parse_pdb(path: str, source: str, *, gene=None, uniprot=None,
               plddt=None, pae=None, meta=None) -> MolStructure:
    """Parse a PDB file into a MolStructure (first model, protein atoms)."""
    pdb = _need("biotite.structure.io.pdb", "mol", "ov.mol")
    f = pdb.PDBFile.read(path)
    atoms = f.get_structure(model=1, extra_fields=["b_factor"])
    # keep amino-acid atoms only
    import biotite.structure as struc
    atoms = atoms[struc.filter_amino_acids(atoms)]
    if plddt is None and source in ("alphafold", "esmfold"):
        # AlphaFold / ESMFold store per-residue pLDDT in the B-factor column
        ca = atoms[atoms.atom_name == "CA"]
        plddt = np.asarray(ca.b_factor, dtype=float)
        # the ESMFold API serves pLDDT on a 0-1 scale; AlphaFold uses
        # 0-100 — rescale so .plddt is always 0-100
        if plddt.size and float(np.nanmax(plddt)) <= 1.0:
            plddt = plddt * 100.0
    return MolStructure(atoms, source=source, path=path, gene=gene,
                        uniprot=uniprot, plddt=plddt, pae=pae, meta=meta)


def _alphafold_record(acc: str) -> dict:
    """Look up an AlphaFold DB prediction record (resolves current URLs)."""
    import requests
    r = requests.get(_AF_API.format(acc=acc), timeout=60)
    if r.status_code == 404:
        raise ValueError(
            f"no AlphaFold DB model for {acc!r} — it may be absent from "
            f"AlphaFold DB. Try source='pdb' or predict_structure().")
    r.raise_for_status()
    records = r.json()
    if not records:
        raise ValueError(f"no AlphaFold DB model for {acc!r}")
    return records[0]


def _load_pae(path: str) -> Optional[np.ndarray]:
    """Parse an AlphaFold DB predicted-aligned-error JSON."""
    try:
        data = json.load(open(path))
    except Exception:  # pragma: no cover
        return None
    obj = data[0] if isinstance(data, list) else data
    for key in ("predicted_aligned_error", "pae", "distance"):
        if key in obj:
            return np.asarray(obj[key], dtype=float)
    return None


# ------------------------------------------------------------------ #
# public                                                              #
# ------------------------------------------------------------------ #


@register_function(
    aliases=["fetch_structure", "蛋白结构", "get_structure", "蛋白质结构",
             "protein_structure", "load_structure"],
    category="mol",
    description=(
        "Fetch a protein 3D structure for an omics target — an experimental "
        "structure from the RCSB PDB or a predicted model from AlphaFold DB. "
        "Accepts a gene symbol, UniProt accession or PDB ID; resolves "
        "gene->UniProt, downloads + caches the file, and returns a "
        "MolStructure carrying per-residue pLDDT confidence and the "
        "predicted-aligned-error matrix for AlphaFold models."
    ),
    examples=[
        "s = ov.mol.fetch_structure('EGFR')          # gene -> AlphaFold model",
        "s = ov.mol.fetch_structure('P00533')        # UniProt accession",
        "s = ov.mol.fetch_structure('1M17', source='pdb')  # experimental",
    ],
    related=["mol.predict_structure", "mol.view", "mol.pockets"],
)
def fetch_structure(query: str, *, source: str = "auto",
                    taxon: int = 9606, dir: str = "./data",
                    verbose: bool = False) -> MolStructure:
    r"""Fetch a protein structure for ``query``.

    Parameters
    ----------
    query
        A gene symbol (e.g. ``'EGFR'``), a UniProt accession
        (``'P00533'``) or a 4-character PDB ID (``'1M17'``).
    source
        ``'auto'`` — AlphaFold DB for gene/UniProt queries, RCSB PDB for
        PDB-ID queries; or force ``'pdb'`` / ``'alphafold'``.
    taxon
        NCBI taxon id for gene resolution (default 9606, human).
    dir
        Directory the structure files are cached in.
    verbose
        Print resolution steps.

    Returns
    -------
    MolStructure
    """
    check_core()
    os.makedirs(dir, exist_ok=True)
    q = query.strip()

    is_pdb_id = bool(_PDB_RE.match(q)) and source != "alphafold"
    if source == "pdb" or (source == "auto" and is_pdb_id):
        pdb_id = q.upper()
        path = _download(_RCSB.format(pdb=pdb_id), os.path.join(dir, f"{pdb_id}.pdb"))
        if verbose:
            print(f"fetched experimental structure {pdb_id} from RCSB PDB")
        return _parse_pdb(path, "pdb", meta={"pdb_id": pdb_id})

    # gene / UniProt -> AlphaFold DB
    gene = None
    if _UNIPROT_RE.match(q):
        acc = q
    else:
        gene, acc = q, _gene_to_uniprot(q, taxon)
        if verbose:
            print(f"resolved gene {gene} -> UniProt {acc}")

    rec = _alphafold_record(acc)
    gene = gene or rec.get("gene")
    model_id = rec.get("modelEntityId", f"AF-{acc}-F1")
    pdb_path = _download(rec["pdbUrl"], os.path.join(dir, f"{model_id}.pdb"))
    pae = None
    try:
        pae_path = _download(rec["paeDocUrl"],
                             os.path.join(dir, f"{model_id}-pae.json"))
        pae = _load_pae(pae_path)
    except Exception:  # pragma: no cover - PAE is best-effort
        if verbose:
            print("PAE JSON unavailable for this model")
    if verbose:
        print(f"fetched AlphaFold model {model_id} "
              f"(v{rec.get('latestVersion', '?')})")
    return _parse_pdb(
        pdb_path, "alphafold", gene=gene, uniprot=acc, pae=pae,
        meta={"model": model_id,
              "version": rec.get("latestVersion"),
              "global_plddt": rec.get("globalMetricValue"),
              "organism": rec.get("organismScientificName")})


@register_function(
    aliases=["predict_structure", "结构预测", "esmfold", "fold_sequence",
             "蛋白结构预测"],
    category="mol",
    description=(
        "Predict a protein 3D structure from its amino-acid sequence with "
        "the ESMFold remote API — for sequences absent from AlphaFold DB "
        "(mutants, non-model species, designed sequences). No local GPU "
        "required. Returns a MolStructure with per-residue pLDDT."
    ),
    examples=[
        "s = ov.mol.predict_structure('MKTAYIAKQR...')",
    ],
    related=["mol.fetch_structure", "mol.view"],
)
def predict_structure(sequence: str, *, engine: str = "esmfold",
                      name: Optional[str] = None, dir: str = "./data",
                      timeout: int = 300, verbose: bool = False) -> MolStructure:
    r"""Predict a structure from a raw amino-acid sequence.

    Parameters
    ----------
    sequence
        One-letter amino-acid sequence (<= ~400 residues for the public
        ESMFold API).
    engine
        Currently only ``'esmfold'`` (the ESM Metagenomic Atlas API).
    name
        Optional name for the cached file.
    dir
        Cache directory.
    timeout
        Request timeout (s).

    Returns
    -------
    MolStructure
        ``.source='esmfold'``, with per-residue pLDDT in ``.plddt``.
    """
    check_core()
    import requests
    if engine != "esmfold":
        raise ValueError(f"engine must be 'esmfold', got {engine!r}")
    seq = "".join(sequence.split()).upper()
    os.makedirs(dir, exist_ok=True)
    name = name or f"esmfold_{abs(hash(seq)) % 10**8}"
    path = os.path.join(dir, f"{name}.pdb")
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        if verbose:
            print(f"folding {len(seq)} residues with the ESMFold API ...")
        r = requests.post(_ESMFOLD, data=seq, timeout=timeout)
        r.raise_for_status()
        with open(path, "w") as fh:
            fh.write(r.text)
    return _parse_pdb(path, "esmfold", meta={"name": name, "engine": "esmfold"})
